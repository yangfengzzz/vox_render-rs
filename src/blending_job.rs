/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use crate::soa_transform::SoaTransform;
use crate::simd_math::SimdFloat4;
use crate::soa_quaternion::SoaQuaternion;
use crate::soa_float::SoaFloat3;

// Defines a layer of blending input data (local space transforms) and
// parameters (weights).
pub struct Layer<'a> {
    // Blending weight of this layer. Negative values are considered as 0.
    // Normalization is performed during the blending stage so weight can be in
    // any range, even though range [0:1] is optimal.
    pub weight: f32,

    // The range [begin,end[ of input layer posture. This buffer expect to store
    // local space transforms, that are usually outputted from a sampling job.
    // This range must be at least as big as the bind pose buffer, even though
    // only the number of transforms defined by the bind pose buffer will be
    // processed.
    pub transform: &'a [SoaTransform],

    // Optional range [begin,end[ of blending weight for each joint in this
    // layer.
    // If both pointers are nullptr (default case) then per joint weight
    // blending is disabled. A valid range is defined as being at least as big
    // as the bind pose buffer, even though only the number of transforms
    // defined by the bind pose buffer will be processed. When a layer doesn't
    // specifies per joint weights, then it is implicitly considered as
    // being 1.f. This default value is a reference value for the normalization
    // process, which implies that the range of values for joint weights should
    // be [0,1]. Negative weight values are considered as 0, but positive ones
    // aren't clamped because they could exceed 1.f if all layers contains valid
    // joint weights.
    pub joint_weights: &'a [SimdFloat4],
}

impl<'a> Layer<'a> {
    // Default constructor, initializes default values.
    pub fn new() -> Layer<'a> {
        return Layer {
            weight: 0.0,
            transform: &[],
            joint_weights: &[],
        };
    }
}

// ozz::animation::BlendingJob is in charge of blending (mixing) multiple poses
// (the result of a sampled animation) according to their respective weight,
// into one output pose.
// The number of transforms/joints blended by the job is defined by the number
// of transforms of the bind pose (note that this is a SoA format). This means
// that all buffers must be at least as big as the bind pose buffer.
// Partial animation blending is supported through optional joint weights that
// can be specified with layers joint_weights buffer. Unspecified joint weights
// are considered as a unit weight of 1.f, allowing to mix full and partial
// blend operations in a single pass.
// The job does not owned any buffers (input/output) and will thus not delete
// them during job's destruction.
pub struct BlendingJob<'a> {
    // The job blends the bind pose to the output when the accumulated weight of
    // all layers is less than this threshold value.
    // Must be greater than 0.f.
    pub threshold: f32,

    // Job input layers, can be empty or nullptr.
    // The range of layers that must be blended.
    pub layers: &'a [Layer<'a>],

    // Job input additive layers, can be empty or nullptr.
    // The range of layers that must be added to the output.
    pub additive_layers: &'a [Layer<'a>],

    // The skeleton bind pose. The size of this buffer defines the number of
    // transforms to blend. This is the reference because this buffer is defined
    // by the skeleton that all the animations belongs to.
    // It is used when the accumulated weight for a bone on all layers is
    // less than the threshold value, in order to fall back on valid transforms.
    pub bind_pose: &'a [SoaTransform],

    // Job output.
    // The range of output transforms to be filled with blended layer
    // transforms during job execution.
    // Must be at least as big as the bind pose buffer, but only the number of
    // transforms defined by the bind pose buffer size will be processed.
    pub output: &'a mut [SoaTransform],
}

impl<'a> BlendingJob<'a> {
    // Default constructor, initializes default values.
    pub fn new() -> BlendingJob<'a> {
        return BlendingJob {
            threshold: 0.1,
            layers: &[],
            additive_layers: &[],
            bind_pose: &[],
            output: &mut [],
        };
    }

    // Validates job parameters.
    // Returns true for a valid job, false otherwise:
    // -if layer range is not valid (can be empty though).
    // -if additive layer range is not valid (can be empty though).
    // -if any layer is not valid.
    // -if output range is not valid.
    // -if any buffer (including layers' content : transform, joint weights...) is
    // smaller than the bind pose buffer.
    // -if the threshold value is less than or equal to 0.f.
    pub fn validate(&self) -> bool {
        // Don't need any early out, as jobs are valid in most of the performance
        // critical cases.
        // Tests are written in multiple lines in order to avoid branches.
        let mut valid = true;

        // Test for valid threshold).
        valid &= self.threshold > 0.0;

        // Test for nullptr begin pointers.
        // Blending layers are mandatory, additive aren't.
        valid &= !self.bind_pose.is_empty();
        valid &= !self.output.is_empty();

        // The bind pose size defines the ranges of transforms to blend, so all
        // other buffers should be bigger.
        let min_range = self.bind_pose.len();
        valid &= self.output.len() >= min_range;

        // Validates layers.
        for layer in self.layers {
            valid &= validate_layer(layer, min_range);
        }

        // Validates additive layers.
        for layer in self.additive_layers {
            valid &= validate_layer(layer, min_range);
        }

        return valid;
    }

    // Runs job's blending task.
    // The job is validated before any operation is performed, see validate() for
    // more details.
    // Returns false if *this job is not valid.
    pub fn run(&'a mut self) -> bool {
        if !self.validate() {
            return false;
        }

        // Initializes blended parameters that are exchanged across blend stages.
        let mut process_args = ProcessArgs::new(self);

        // Blends all layers to the job output buffers.
        blend_layers(&mut process_args);

        // Applies bind pose.
        blend_bind_pose(&mut process_args);

        // Normalizes output.
        normalize(&mut process_args);

        // Process additive blending.
        add_layers(&mut process_args);

        return true;
    }
}

fn validate_layer(_layer: &Layer, _min_range: usize) -> bool {
    let mut valid = true;

    // Tests transforms validity.
    valid &= _layer.transform.len() >= _min_range;

    // Joint weights are optional.
    if !_layer.joint_weights.is_empty() {
        valid &= _layer.joint_weights.len() >= _min_range;
    } else {
        valid &= _layer.joint_weights.is_empty();
    }
    return valid;
}

// Defines parameters that are passed through blending stages.
struct ProcessArgs<'a> {
    // Allocates enough space to store a accumulated weights per-joint.
    // It will be initialized by the first pass processed, if any.
    // This is quite big for a stack allocation (4 byte * maximum number of
    // joints). This is one of the reasons why the number of joints is limited
    // by the API.
    // Note that this array is used with SoA data.
    // This is the first argument in order to avoid wasting too much space with
    // alignment padding.
    accumulated_weights: [SimdFloat4; crate::skeleton::Constants::KMaxSoAJoints as usize],

    // The job to process.
    job: &'a mut BlendingJob<'a>,

    // The number of transforms to process as defined by the size of the bind
    // pose.
    num_soa_joints: usize,

    // Number of processed blended passes (excluding passes with a weight <= 0.f),
    // including partial passes.
    num_passes: i32,

    // Number of processed partial blending passes (aka with a weight per-joint).
    num_partial_passes: i32,

    // The accumulated weight of all layers.
    accumulated_weight: f32,
}

impl<'a> ProcessArgs<'a> {
    fn new(_job: &'a mut BlendingJob<'a>) -> ProcessArgs<'a> {
        let args = ProcessArgs {
            accumulated_weights: [SimdFloat4::zero(); crate::skeleton::Constants::KMaxSoAJoints as usize],
            num_soa_joints: _job.bind_pose.len(),
            job: _job,
            num_passes: 0,
            num_partial_passes: 0,
            accumulated_weight: 0.0,
        };
        // The range of all buffers has already been validated.
        debug_assert!(args.job.output.len() >= args.num_soa_joints);
        debug_assert!(args.accumulated_weights.len() >= args.num_soa_joints);
        return args;
    }
}

// Blends all layers of the job to its output.
fn blend_layers<'a, 'b>(_args: &'b mut ProcessArgs<'a>) {
    // Iterates through all layers and blend them to the output.
    for layer in _args.job.layers {
        // Asserts buffer sizes, which must never fail as it has been validated.
        debug_assert!(layer.transform.len() >= _args.num_soa_joints);
        debug_assert!(layer.joint_weights.is_empty() ||
            (layer.joint_weights.len() >= _args.num_soa_joints));

        // Skip irrelevant layers.
        if layer.weight <= 0.0 {
            continue;
        }

        // Accumulates global weights.
        _args.accumulated_weight += layer.weight;
        let layer_weight = SimdFloat4::load1(layer.weight);

        if !layer.joint_weights.is_empty() {
            // This layer has per-joint weights.
            _args.num_partial_passes += 1;

            if _args.num_passes == 0 {
                for i in 0.._args.num_soa_joints {
                    let src = &layer.transform[i];
                    let dest = &mut _args.job.output[i];
                    let weight = layer_weight * layer.joint_weights[i].max0();
                    _args.accumulated_weights[i] = weight;
                    ozz_blend_1st_pass(src, weight, dest);
                }
            } else {
                for i in 0.._args.num_soa_joints {
                    let src = &layer.transform[i];
                    let dest = &mut _args.job.output[i];
                    let weight = layer_weight * layer.joint_weights[i].max0();
                    _args.accumulated_weights[i] = _args.accumulated_weights[i] + weight;
                    ozz_blend_n_pass(src, weight, dest);
                }
            }
        } else {
            // This is a full layer.
            if _args.num_passes == 0 {
                for i in 0.._args.num_soa_joints {
                    let src = &layer.transform[i];
                    let dest = &mut _args.job.output[i];
                    _args.accumulated_weights[i] = layer_weight;
                    ozz_blend_1st_pass(src, layer_weight, dest);
                }
            } else {
                for i in 0.._args.num_soa_joints {
                    let src = &layer.transform[i];
                    let dest = &mut _args.job.output[i];
                    _args.accumulated_weights[i] = _args.accumulated_weights[i] + layer_weight;
                    ozz_blend_n_pass(src, layer_weight, dest);
                }
            }
        }
        // One more pass blended.
        _args.num_passes += 1;
    }
}

// Blends bind pose to the output if accumulated weight is less than the
// threshold value.
fn blend_bind_pose<'a, 'b>(_args: &'b mut ProcessArgs<'a>) {
    // Asserts buffer sizes, which must never fail as it has been validated.
    debug_assert!(_args.job.bind_pose.len() >= _args.num_soa_joints);

    if _args.num_partial_passes == 0 {
        // No partial blending pass detected, threshold can be tested globally.
        let bp_weight = _args.job.threshold - _args.accumulated_weight;

        if bp_weight > 0.0 {  // The bind-pose is needed if it has a weight.
            if _args.num_passes == 0 {
                // Strictly copying bind-pose.
                _args.accumulated_weight = 1.0;
                for i in 0.._args.num_soa_joints {
                    _args.job.output[i] = _args.job.bind_pose[i];
                }
            } else {
                // Updates global accumulated weight, but not per-joint weight any more
                // because normalization stage will be global also.
                _args.accumulated_weight = _args.job.threshold;

                let simd_bp_weight = SimdFloat4::load1(bp_weight);

                for i in 0.._args.num_soa_joints {
                    let src = &_args.job.bind_pose[i];
                    let dest = &mut _args.job.output[i];
                    ozz_blend_n_pass(src, simd_bp_weight, dest);
                }
            }
        }
    } else {
        // Blending passes contain partial blending, threshold must be tested for
        // each joint.
        let threshold = SimdFloat4::load1(_args.job.threshold);

        // There's been at least 1 pass as num_partial_passes != 0.
        debug_assert!(_args.num_passes != 0);

        for i in 0.._args.num_soa_joints {
            let src = &_args.job.bind_pose[i];
            let dest = &mut _args.job.output[i];
            let bp_weight = (threshold - _args.accumulated_weights[i]).max0();
            _args.accumulated_weights[i] = threshold.max(_args.accumulated_weights[i]);
            ozz_blend_n_pass(src, bp_weight, dest);
        }
    }
}

// Normalizes output rotations. Quaternion length cannot be zero as opposed
// quaternions have been fixed up during blending passes.
// Translations and scales are already normalized because weights were
// pre-multiplied by the normalization ratio.
fn normalize<'a, 'b>(_args: &'b mut ProcessArgs<'a>) {
    if _args.num_partial_passes == 0 {
        // Normalization of a non-partial blending requires to apply the same
        // division to all joints.
        let ratio = SimdFloat4::load1(1.0 / _args.accumulated_weight);
        for i in 0.._args.num_soa_joints {
            let dest = &mut _args.job.output[i];
            dest.rotation = dest.rotation.normalize_est();
            dest.translation = dest.translation * ratio;
            dest.scale = dest.scale * ratio;
        }
    } else {
        // Partial blending normalization requires to compute the divider per-joint.
        let one = SimdFloat4::one();
        for i in 0.._args.num_soa_joints {
            let ratio = one / _args.accumulated_weights[i];
            let dest = &mut _args.job.output[i];
            dest.rotation = dest.rotation.normalize_est();
            dest.translation = dest.translation * ratio;
            dest.scale = dest.scale * ratio;
        }
    }
}

// Process additive blending pass.
fn add_layers<'a, 'b>(_args: &'b mut ProcessArgs<'a>) {
    // Iterates through all layers and blend them to the output.
    for layer in _args.job.additive_layers {
        // Asserts buffer sizes, which must never fail as it has been validated.
        debug_assert!(layer.transform.len() >= _args.num_soa_joints);
        debug_assert!(layer.joint_weights.is_empty() ||
            (layer.joint_weights.len() >= _args.num_soa_joints));

        // Prepares constants.
        let one = SimdFloat4::one();

        if layer.weight > 0.0 {
            // Weight is positive, need to perform additive blending.
            let layer_weight = SimdFloat4::load1(layer.weight);

            if !layer.joint_weights.is_empty() {
                // This layer has per-joint weights.
                for i in 0.._args.num_soa_joints {
                    let src = &layer.transform[i];
                    let dest = &mut _args.job.output[i];
                    let weight = layer_weight * layer.joint_weights[i].max0();
                    let one_minus_weight = one - weight;
                    let one_minus_weight_f3 = SoaFloat3::load(
                        one_minus_weight, one_minus_weight, one_minus_weight);
                    ozz_add_pass(src, weight, dest, &one_minus_weight_f3, &one);
                }
            } else {
                // This is a full layer.
                let one_minus_weight = one - layer_weight;
                let one_minus_weight_f3 = SoaFloat3::load(
                    one_minus_weight, one_minus_weight, one_minus_weight);

                for i in 0.._args.num_soa_joints {
                    let src = &layer.transform[i];
                    let dest = &mut _args.job.output[i];
                    ozz_add_pass(src, layer_weight, dest, &one_minus_weight_f3, &one);
                }
            }
        } else if layer.weight < 0.0 {
            // Weight is negative, need to perform subtractive blending.
            let layer_weight = SimdFloat4::load1(-layer.weight);

            if !layer.joint_weights.is_empty() {
                // This layer has per-joint weights.
                for i in 0.._args.num_soa_joints {
                    let src = &layer.transform[i];
                    let dest = &mut _args.job.output[i];
                    let weight = layer_weight * layer.joint_weights[i].max0();
                    let one_minus_weight = one - weight;
                    ozz_sub_pass(src, weight, dest, &one_minus_weight, &one);
                }
            } else {
                // This is a full layer.
                let one_minus_weight = one - layer_weight;
                for i in 0.._args.num_soa_joints {
                    let src = &layer.transform[i];
                    let dest = &mut _args.job.output[i];
                    ozz_sub_pass(src, layer_weight, dest, &one_minus_weight, &one);
                }
            }
        } else {
            // Skip layer as its weight is 0.
        }
    }
}

// defines the process of blending the 1st pass.
fn ozz_blend_1st_pass(_in: &SoaTransform, _simd_weight: SimdFloat4, _out: &mut SoaTransform) {
    _out.translation = _in.translation * _simd_weight;
    _out.rotation = _in.rotation * _simd_weight;
    _out.scale = _in.scale * _simd_weight;
}

// defines the process of blending any pass but the first.
fn ozz_blend_n_pass(_in: &SoaTransform, _simd_weight: SimdFloat4, _out: &mut SoaTransform) {
    /* Blends translation. */
    _out.translation = _out.translation + _in.translation * _simd_weight;
    /* Blends rotations, negates opposed quaternions to be sure to choose*/
    /* the shortest path between the two.*/
    let sign = _out.rotation.dot(&_in.rotation).sign();
    let rotation = SoaQuaternion::load(_in.rotation.x.xor_fi(sign), _in.rotation.y.xor_fi(sign),
                                       _in.rotation.z.xor_fi(sign), _in.rotation.w.xor_fi(sign));
    _out.rotation = _out.rotation + rotation * _simd_weight;
    /* Blends scales.*/
    _out.scale = _out.scale + _in.scale * _simd_weight;
}

// Macro that defines the process of adding a pass.
fn ozz_add_pass(_in: &SoaTransform, _simd_weight: SimdFloat4, _out: &mut SoaTransform,
                one_minus_weight_f3: &SoaFloat3, one: &SimdFloat4) {
    _out.translation = _out.translation + _in.translation * _simd_weight;
    /* Interpolate quaternion between identity and src.rotation.*/
    /* Quaternion sign is fixed up, so that lerp takes the shortest path.*/
    let sign = _in.rotation.w.sign();
    let rotation = SoaQuaternion::load(
        _in.rotation.x.xor_fi(sign), _in.rotation.y.xor_fi(sign),
        _in.rotation.z.xor_fi(sign), _in.rotation.w.xor_fi(sign));
    let interp_quat = SoaQuaternion::load(
        rotation.x * _simd_weight, rotation.y * _simd_weight,
        rotation.z * _simd_weight, (rotation.w - *one) * _simd_weight + *one);
    _out.rotation = interp_quat.normalize_est() * _out.rotation;
    _out.scale =
        _out.scale * (*one_minus_weight_f3 + (_in.scale * _simd_weight));
}

// Macro that defines the process of subtracting a pass.
fn ozz_sub_pass(_in: &SoaTransform, _simd_weight: SimdFloat4, _out: &mut SoaTransform,
                one_minus_weight: &SimdFloat4, one: &SimdFloat4) {
    _out.translation = _out.translation - _in.translation * _simd_weight;
    /* Interpolate quaternion between identity and src.rotation.*/
    /* Quaternion sign is fixed up, so that lerp takes the shortest path.*/
    let sign = _in.rotation.w.sign();
    let rotation = SoaQuaternion::load(
        _in.rotation.x.xor_fi(sign), _in.rotation.y.xor_fi(sign),
        _in.rotation.z.xor_fi(sign), _in.rotation.w.xor_fi(sign));
    let interp_quat = SoaQuaternion::load(
        rotation.x * _simd_weight, rotation.y * _simd_weight,
        rotation.z * _simd_weight, (rotation.w - *one) * _simd_weight + *one);
    _out.rotation = interp_quat.normalize_est().conjugate() * _out.rotation;
    let rcp_scale = SoaFloat3::load(
        _in.scale.x.madd(_simd_weight, *one_minus_weight).rcp_est(),
        _in.scale.y.madd(_simd_weight, *one_minus_weight).rcp_est(),
        _in.scale.z.madd(_simd_weight, *one_minus_weight).rcp_est());
    _out.scale = _out.scale * rcp_scale;
}

//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------
#[cfg(test)]
mod blending_job{
    use crate::soa_transform::SoaTransform;
    use crate::simd_math::SimdFloat4;
    use crate::blending_job::{BlendingJob, Layer};

    #[test]
    fn job_validity() {
        let identity = SoaTransform::identity();
        let zero = SimdFloat4::zero();
        let mut layers = [Layer::new(), Layer::new()];
        let bind_poses = [identity, identity, identity];
        let input_transforms = [identity, identity, identity];
        let mut output_transforms = [identity, identity, identity];
        let joint_weights = [zero, zero, zero];

        layers[0].transform = &input_transforms[..];
        layers[1].transform = &input_transforms[0..2];

        {  // Empty/default job.
            let mut job = BlendingJob::new();
            assert_eq!(job.validate(), false);
            assert_eq!(job.run(), false);
        }

        {  // Invalid output.
            let mut job = BlendingJob::new();
            job.layers = &layers[0..2];
            job.bind_pose = &bind_poses[0..2];
            assert_eq!(job.validate(), false);
            assert_eq!(job.run(), false);
        }
        {  // Layers are optional.
            let mut job = BlendingJob::new();
            job.bind_pose = &bind_poses[0..2];
            job.output = &mut output_transforms[0..2];
            assert_eq!(job.validate(), true);
            assert_eq!(job.run(), true);
        }
        {  // Invalid bind pose.
            let mut job = BlendingJob::new();
            job.layers = &layers[0..2];
            job.output = &mut output_transforms[0..2];
            assert_eq!(job.validate(), false);
            assert_eq!(job.run(), false);
        }
        {  // Invalid layer input range, too small.
            let mut invalid_layers = [Layer::new(), Layer::new()];
            invalid_layers[0].transform = &input_transforms[0..1];
            invalid_layers[1].transform = &input_transforms[0..2];

            let mut job = BlendingJob::new();
            job.layers = &invalid_layers[..];
            job.bind_pose = &bind_poses[0..2];
            job.output = &mut output_transforms[0..2];
            assert_eq!(job.validate(), false);
            assert_eq!(job.run(), false);
        }
        {  // Invalid output range, smaller output.
            let mut job = BlendingJob::new();
            job.layers = &layers[0..2];
            job.bind_pose = &bind_poses[0..2];
            job.output = &mut output_transforms[0..1];
            assert_eq!(job.validate(), false);
            assert_eq!(job.run(), false);
        }

        {  // Invalid smaller input.
            let mut job = BlendingJob::new();
            job.layers = &layers[0..2];
            job.bind_pose = &bind_poses[0..3];
            job.output = &mut output_transforms[0..3];
            assert_eq!(job.validate(), false);
            assert_eq!(job.run(), false);
        }

        {  // Invalid threshold.
            let mut job = BlendingJob::new();
            job.threshold = 0.0;
            job.layers = &layers[0..2];
            job.bind_pose = &bind_poses[0..2];
            job.output = &mut output_transforms[0..2];
            assert_eq!(job.validate(), false);
            assert_eq!(job.run(), false);
        }

        {  // Invalid joint weights range.
            layers[0].joint_weights = &joint_weights[0..1];

            let mut job = BlendingJob::new();
            job.layers = &layers[0..2];
            job.bind_pose = &bind_poses[0..2];
            job.output = &mut output_transforms[0..2];
            assert_eq!(job.validate(), false);
            assert_eq!(job.run(), false);
        }

        // {  // Valid job.
        //     layers[0].joint_weights = {nullptr, nullptr};
        //
        //     let mut job = BlendingJob::new();
        //     job.layers = &layers[0..2];
        //     job.bind_pose = &bind_poses[0..2];
        //     job.output = &mut output_transforms[0..2];
        //     assert_eq!(job.validate(), true);
        //     assert_eq!(job.run(), true);
        // }

        {  // Valid joint weights range.
            layers[0].joint_weights = &joint_weights[0..2];

            let mut job = BlendingJob::new();
            job.layers = &layers[0..2];
            job.bind_pose = &bind_poses[0..2];
            job.output = &mut output_transforms[0..2];
            assert_eq!(job.validate(), true);
            assert_eq!(job.run(), true);
        }

        {  // Valid job, bigger output.
            layers[0].joint_weights = &joint_weights[0..2];

            let mut job = BlendingJob::new();
            job.layers = &layers[0..2];
            job.bind_pose = &bind_poses[0..2];
            job.output = &mut output_transforms[0..3];
            assert_eq!(job.validate(), true);
            assert_eq!(job.run(), true);
        }

        {  // Valid no layers.
            let mut job = BlendingJob::new();
            job.bind_pose = &bind_poses[0..2];
            job.output = &mut output_transforms[0..2];
            assert_eq!(job.validate(), true);
            assert_eq!(job.run(), true);
        }
    }
}
