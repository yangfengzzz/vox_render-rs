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
// are considered as a unit weight of 1.0, allowing to mix full and partial
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

    // Number of processed blended passes (excluding passes with a weight <= 0.0),
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
mod blending_job {
    use crate::soa_transform::SoaTransform;
    use crate::simd_math::SimdFloat4;
    use crate::blending_job::{BlendingJob, Layer};
    use crate::soa_float::SoaFloat3;
    use crate::math_test_helper::*;
    use crate::simd_math::*;
    use crate::*;
    use crate::soa_quaternion::SoaQuaternion;

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

        {  // empty/default job.
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

    #[test]
    fn job_validity_additive() {
        let identity = SoaTransform::identity();
        let zero = SimdFloat4::zero();
        let mut layers = [Layer::new(), Layer::new()];
        let mut additive_layers = [Layer::new(), Layer::new()];

        let bind_poses = [identity, identity, identity];
        let input_transforms = [identity, identity, identity];
        let mut output_transforms = [identity, identity, identity];
        let joint_weights = [zero, zero, zero];

        layers[0].transform = &input_transforms[..];
        layers[1].transform = &input_transforms[..];

        additive_layers[0].transform = &input_transforms[..];
        additive_layers[1].transform = &input_transforms[..];

        {  // Valid additive job, no normal blending.
            let mut job = BlendingJob::new();
            job.additive_layers = &additive_layers[..];
            job.bind_pose = &bind_poses[..];
            job.output = &mut output_transforms[..];
            assert_eq!(job.validate(), true);
            assert_eq!(job.run(), true);
        }

        {  // Valid additive job, with normal blending also.

            let mut job = BlendingJob::new();
            job.layers = &layers[..];
            job.additive_layers = &additive_layers[..];
            job.bind_pose = &bind_poses[..];
            job.output = &mut output_transforms[..];
            assert_eq!(job.validate(), true);
            assert_eq!(job.run(), true);
        }

        {  // Invalid layer input range, too small.
            let mut invalid_layers = [Layer::new(), Layer::new()];
            invalid_layers[0].transform = &input_transforms[0..1];
            invalid_layers[1].transform = &input_transforms[0..2];

            let mut job = BlendingJob::new();
            job.layers = &layers[..];
            job.additive_layers = &invalid_layers[..];
            job.bind_pose = &bind_poses[0..2];
            job.output = &mut output_transforms[0..2];
            assert_eq!(job.validate(), false);
            assert_eq!(job.run(), false);
        }

        {  // Valid additive job, with per-joint weights.
            layers[0].joint_weights = &joint_weights[0..2];

            let mut job = BlendingJob::new();
            job.additive_layers = &additive_layers[..];
            job.bind_pose = &bind_poses[0..2];
            job.output = &mut output_transforms[0..2];
            assert_eq!(job.validate(), true);
            assert_eq!(job.run(), true);
        }
    }

    #[test]
    fn empty() {
        let identity = SoaTransform::identity();

        // Initialize bind pose.
        let mut bind_poses = [identity, identity];
        bind_poses[0].translation = SoaFloat3::load(
            SimdFloat4::load(0.0, 1.0, 2.0, 3.0),
            SimdFloat4::load(4.0, 5.0, 6.0, 7.0),
            SimdFloat4::load(8.0, 9.0, 10.0, 11.0));
        bind_poses[0].scale = SoaFloat3::load(
            SimdFloat4::load(0.0, 10.0, 20.0, 30.0),
            SimdFloat4::load(40.0, 50.0, 60.0, 70.0),
            SimdFloat4::load(80.0, 90.0, 100.0, 110.0));
        bind_poses[1].translation = bind_poses[0].translation *
            SimdFloat4::load(2.0, 2.0, 2.0, 2.0);
        bind_poses[1].scale =
            bind_poses[0].scale * SimdFloat4::load(2.0, 2.0, 2.0, 2.0);

        let mut job = BlendingJob::new();
        job.bind_pose = &bind_poses;
        let mut output_transforms = [SoaTransform::identity(); 2];
        job.output = &mut output_transforms;

        assert_eq!(job.validate(), true);
        assert_eq!(job.run(), true);

        expect_soa_float3_eq!(output_transforms[0].translation, 0.0, 1.0, 2.0, 3.0, 4.0,
                            5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0);
        expect_soa_float3_eq!(output_transforms[0].scale, 0.0, 10.0, 20.0, 30.0, 40.0,
                            50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0);
        expect_soa_float3_eq!(output_transforms[1].translation, 0.0, 2.0, 4.0, 6.0, 8.0,
                            10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0);
        expect_soa_float3_eq!(output_transforms[1].scale, 0.0, 20.0, 40.0, 60.0, 80.0,
                            100.0, 120.0, 140.0, 160.0, 180.0, 200.0, 220.0);
    }

    #[test]
    fn weight() {
        let identity = SoaTransform::identity();

        // Initialize inputs.
        let mut input_transforms = [[identity, identity],
            [identity, identity]];
        input_transforms[0][0].translation = SoaFloat3::load(
            SimdFloat4::load(0.0, 1.0, 2.0, 3.0),
            SimdFloat4::load(4.0, 5.0, 6.0, 7.0),
            SimdFloat4::load(8.0, 9.0, 10.0, 11.0));
        input_transforms[0][1].translation = SoaFloat3::load(
            SimdFloat4::load(12.0, 13.0, 14.0, 15.0),
            SimdFloat4::load(16.0, 17.0, 18.0, 19.0),
            SimdFloat4::load(20.0, 21.0, 22.0, 23.0));
        input_transforms[1][0].translation = -input_transforms[0][0].translation;
        input_transforms[1][1].translation = -input_transforms[0][1].translation;

        // Initialize bind pose.
        let mut bind_poses = [identity, identity];
        bind_poses[0].scale = SoaFloat3::load(
            SimdFloat4::load(0.0, 1.0, 2.0, 3.0),
            SimdFloat4::load(4.0, 5.0, 6.0, 7.0),
            SimdFloat4::load(8.0, 9.0, 10.0, 11.0));
        bind_poses[1].scale =
            bind_poses[0].scale * SimdFloat4::load(2.0, 2.0, 2.0, 2.0);

        {
            let mut layers = [Layer::new(), Layer::new()];
            layers[0].transform = &input_transforms[0];
            layers[1].transform = &input_transforms[1];

            let mut output_transforms = [SoaTransform::identity(); 2];

            {
                // Weight 0 (a bit less must give the same result) for the first layer,
                // 1 for the second.
                layers[0].weight = -0.07;
                layers[1].weight = 1.0;

                let mut job = BlendingJob::new();
                job.layers = &layers;
                job.bind_pose = &bind_poses;
                job.output = &mut output_transforms[..];

                assert_eq!(job.run(), true);

                expect_soa_float3_eq!(output_transforms[0].translation, -0.0, -1.0, -2.0,
                                -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0, -11.0);
                expect_soa_float3_eq!(output_transforms[0].scale, 1.0, 1.0, 1.0, 1.0, 1.0,
                                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
                expect_soa_float3_eq!(output_transforms[1].translation, -12.0, -13.0, -14.0,
                                -15.0, -16.0, -17.0, -18.0, -19.0, -20.0, -21.0, -22.0,
                                -23.0);
                expect_soa_float3_eq!(output_transforms[1].scale, 1.0, 1.0, 1.0, 1.0, 1.0,
                                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
            }

            {
                // Weight 1 for the first layer, 0 for the second.
                layers[0].weight = 1.0;
                layers[1].weight = 1e-27;  // Very low weight value.

                let mut job = BlendingJob::new();
                job.layers = &layers;
                job.bind_pose = &bind_poses;
                job.output = &mut output_transforms[..];

                assert_eq!(job.run(), true);

                expect_soa_float3_eq!(output_transforms[0].translation, 0.0, 1.0, 2.0, 3.0,
                                4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0);
                expect_soa_float3_eq!(output_transforms[0].scale, 1.0, 1.0, 1.0, 1.0, 1.0,
                                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
                expect_soa_float3_eq!(output_transforms[1].translation, 12.0, 13.0, 14.0,
                                15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0);
                expect_soa_float3_eq!(output_transforms[1].scale, 1.0, 1.0, 1.0, 1.0, 1.0,
                                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
            }

            {
                // Weight .5 for both layers.
                layers[0].weight = 0.5;
                layers[1].weight = 0.5;

                let mut job = BlendingJob::new();
                job.layers = &layers;
                job.bind_pose = &bind_poses;
                job.output = &mut output_transforms[..];

                assert_eq!(job.run(), true);

                expect_soa_float3_eq!(output_transforms[0].translation, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
                expect_soa_float3_eq!(output_transforms[0].scale, 1.0, 1.0, 1.0, 1.0, 1.0,
                                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
                expect_soa_float3_eq!(output_transforms[1].translation, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
                expect_soa_float3_eq!(output_transforms[1].scale, 1.0, 1.0, 1.0, 1.0, 1.0,
                                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
            }
        }
    }

    #[test]
    fn joint_weights() {
        let identity = SoaTransform::identity();

        // Initialize inputs.
        let mut input_transforms = [[identity, identity],
            [identity, identity]];
        input_transforms[0][0].translation = SoaFloat3::load(
            SimdFloat4::load(0.0, 1.0, 2.0, 3.0),
            SimdFloat4::load(4.0, 5.0, 6.0, 7.0),
            SimdFloat4::load(8.0, 9.0, 10.0, 11.0));
        input_transforms[0][1].translation = SoaFloat3::load(
            SimdFloat4::load(12.0, 13.0, 14.0, 15.0),
            SimdFloat4::load(16.0, 17.0, 18.0, 19.0),
            SimdFloat4::load(20.0, 21.0, 22.0, 23.0));
        input_transforms[1][0].translation = -input_transforms[0][0].translation;
        input_transforms[1][1].translation = -input_transforms[0][1].translation;
        let joint_weights = [
            [SimdFloat4::load(1.0, 1.0, 0.0, 0.0),
                SimdFloat4::load(1.0, 0.0, 1.0, 1.0)],
            [SimdFloat4::load(1.0, 1.0, 1.0, 0.0),
                SimdFloat4::load(0.0, 1.0, 1.0, 1.0)]];
        // Initialize bind pose.
        let mut bind_poses = [identity, identity];
        bind_poses[0].translation = SoaFloat3::load(
            SimdFloat4::load(10.0, 11.0, 12.0, 13.0),
            SimdFloat4::load(14.0, 15.0, 16.0, 17.0),
            SimdFloat4::load(18.0, 19.0, 20.0, 21.0));
        bind_poses[0].scale = SoaFloat3::load(
            SimdFloat4::load(0.0, 1.0, 2.0, 3.0),
            SimdFloat4::load(4.0, 5.0, 6.0, 7.0),
            SimdFloat4::load(8.0, 9.0, 10.0, 11.0));
        bind_poses[1].scale =
            bind_poses[0].scale * SimdFloat4::load(2.0, 2.0, 2.0, 2.0);

        let mut layers = [Layer::new(), Layer::new()];
        layers[0].transform = &input_transforms[0];
        layers[0].joint_weights = &joint_weights[0];
        layers[1].transform = &input_transforms[1];
        layers[1].joint_weights = &joint_weights[1];

        {  // Weight .5 for both layers.
            let mut output_transforms = [SoaTransform::identity(); 3];

            layers[0].weight = 0.5;
            layers[1].weight = 0.5;

            let mut job = BlendingJob::new();
            job.layers = &layers;
            job.bind_pose = &bind_poses;
            job.output = &mut output_transforms;

            assert_eq!(job.run(), true);

            expect_soa_float3_eq!(output_transforms[0].translation, 0.0, 0.0, -2.0, 13.0,
                                0.0, 0.0, -6.0, 17.0, 0.0, 0.0, -10.0, 21.0);
            expect_soa_float3_eq!(output_transforms[0].scale, 1.0, 1.0, 1.0, 3.0, 1.0,
                                1.0, 1.0, 7.0, 1.0, 1.0, 1.0, 11.0);
            expect_soa_float3_eq!(output_transforms[1].translation, 12.0, -13.0, 0.0, 0.0,
                                16.0, -17.0, 0.0, 0.0, 20.0, -21.0, 0.0, 0.0);
            expect_soa_float3_eq!(output_transforms[1].scale, 1.0, 1.0, 1.0, 1.0, 1.0,
                                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        }
        {  // Null weight for the first layer.
            let mut output_transforms = [SoaTransform::identity(); 2];

            layers[0].weight = 0.0;
            layers[1].weight = 1.0;

            let mut job = BlendingJob::new();
            job.layers = &layers;
            job.bind_pose = &bind_poses;
            job.output = &mut output_transforms;

            assert_eq!(job.run(), true);

            expect_soa_float3_eq!(output_transforms[0].translation, -0.0, -1.0, -2.0,
                                13.0, -4.0, -5.0, -6.0, 17.0, -8.0, -9.0, -10.0, 21.0);
            expect_soa_float3_eq!(output_transforms[0].scale, 1.0, 1.0, 1.0, 3.0, 1.0,
                                1.0, 1.0, 7.0, 1.0, 1.0, 1.0, 11.0);
            expect_soa_float3_eq!(output_transforms[1].translation, 0.0, -13.0, -14.0,
                                -15.0, 0.0, -17.0, -18.0, -19.0, 0.0, -21.0, -22.0,
                                -23.0);
            expect_soa_float3_eq!(output_transforms[1].scale, 0.0, 1.0, 1.0, 1.0, 8.0,
                                1.0, 1.0, 1.0, 16.0, 1.0, 1.0, 1.0);
        }
    }

    #[test]
    fn normalize() {
        let identity = SoaTransform::identity();

        // Initialize inputs.
        let mut input_transforms = [[identity], [identity]];

        // Initialize bind pose.
        let mut bind_poses = [identity];
        bind_poses[0].scale = SoaFloat3::load(
            SimdFloat4::load(0.0, 1.0, 2.0, 3.0),
            SimdFloat4::load(4.0, 5.0, 6.0, 7.0),
            SimdFloat4::load(8.0, 9.0, 10.0, 11.0));

        input_transforms[0][0].rotation = SoaQuaternion::load(
            SimdFloat4::load(0.70710677, 0.0, 0.0, 0.382683432),
            SimdFloat4::load(0.0, 0.0, 0.70710677, 0.0),
            SimdFloat4::load(0.0, 0.0, 0.0, 0.0),
            SimdFloat4::load(0.70710677, 1.0, 0.70710677, 0.9238795));
        input_transforms[1][0].rotation = SoaQuaternion::load(
            SimdFloat4::load(0.0, 0.70710677, -0.70710677, -0.382683432),
            SimdFloat4::load(0.0, 0.0, 0.0, 0.0),
            SimdFloat4::load(0.0, 0.0, -0.70710677, 0.0),
            SimdFloat4::load(1.0, 0.70710677, 0.0, -0.9238795));

        {  // Un-normalized weights < 1.f.
            input_transforms[0][0].translation = SoaFloat3::load(
                SimdFloat4::load(2.0, 3.0, 4.0, 5.0),
                SimdFloat4::load(6.0, 7.0, 8.0, 9.0),
                SimdFloat4::load(10.0, 11.0, 12.0, 13.0));
            input_transforms[1][0].translation = SoaFloat3::load(
                SimdFloat4::load(3.0, 4.0, 5.0, 6.0),
                SimdFloat4::load(7.0, 8.0, 9.0, 10.0),
                SimdFloat4::load(11.0, 12.0, 13.0, 14.0));

            let mut layers = [Layer::new(), Layer::new()];
            layers[0].weight = 0.2;
            layers[0].transform = &input_transforms[0];
            layers[1].weight = 0.3;
            layers[1].transform = &input_transforms[1];

            let mut output_transforms = [SoaTransform::identity(); 1];

            let mut job = BlendingJob::new();
            job.layers = &layers;
            job.bind_pose = &bind_poses;
            job.output = &mut output_transforms;

            assert_eq!(job.run(), true);

            expect_soa_float3_eq!(output_transforms[0].translation, 2.6, 3.6, 4.6,
                                5.6, 6.6, 7.6, 8.6, 9.6, 10.6, 11.6, 12.6,
                                13.6);
            expect_soa_quaternion_eq_est!(output_transforms[0].rotation, 0.30507791,
                                        0.45761687, -0.58843851, 0.38268352, 0.0, 0.0,
                                        0.39229235, 0.0, 0.0, 0.0, -0.58843851, 0.0,
                                        0.95224595, 0.88906217, 0.39229235, 0.92387962);
            assert_eq!(output_transforms[0].rotation.is_normalized_est().are_all_true(), true);
            expect_soa_float3_eq!(output_transforms[0].scale, 1.0, 1.0, 1.0, 1.0, 1.0,
                                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        }
        {  // Un-normalized weights > 1.f.
            input_transforms[0][0].translation = SoaFloat3::load(
                SimdFloat4::load(5.0, 10.0, 15.0, 20.0),
                SimdFloat4::load(25.0, 30.0, 35.0, 40.0),
                SimdFloat4::load(45.0, 50.0, 55.0, 60.0));
            input_transforms[1][0].translation = SoaFloat3::load(
                SimdFloat4::load(10.0, 15.0, 20.0, 25.0),
                SimdFloat4::load(30.0, 35.0, 40.0, 45.0),
                SimdFloat4::load(50.0, 55.0, 60.0, 65.0));

            let mut layers = [Layer::new(), Layer::new()];
            layers[0].weight = 2.0;
            layers[0].transform = &input_transforms[0];
            layers[1].weight = 3.0;
            layers[1].transform = &input_transforms[1];

            let mut output_transforms = [SoaTransform::identity(); 1];

            let mut job = BlendingJob::new();
            job.layers = &layers;
            job.bind_pose = &bind_poses;
            job.output = &mut output_transforms;

            assert_eq!(job.run(), true);

            expect_soa_float3_eq!(output_transforms[0].translation, 8.0, 13.0, 18.0, 23.0,
                                28.0, 33.0, 38.0, 43.0, 48.0, 53.0, 58.0, 63.0);
            expect_soa_quaternion_eq_est!(output_transforms[0].rotation, 0.30507791,
                                        0.45761687, -0.58843851, 0.38268352, 0.0, 0.0,
                                        0.39229235, 0.0, 0.0, 0.0, -0.58843851, 0.0,
                                        0.95224595, 0.88906217, 0.39229235, 0.92387962);
            assert_eq!(output_transforms[0].rotation.is_normalized_est().are_all_true(), true);
            expect_soa_float3_eq!(output_transforms[0].scale, 1.0, 1.0, 1.0, 1.0, 1.0,
                                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        }
        {  // Un-normalized weights > 1.0, with per-joint weights.
            input_transforms[0][0].translation = SoaFloat3::load(
                SimdFloat4::load(5.0, 10.0, 15.0, 20.0),
                SimdFloat4::load(25.0, 30.0, 35.0, 40.0),
                SimdFloat4::load(45.0, 50.0, 55.0, 60.0));
            input_transforms[1][0].translation = SoaFloat3::load(
                SimdFloat4::load(10.0, 15.0, 20.0, 25.0),
                SimdFloat4::load(30.0, 35.0, 40.0, 45.0),
                SimdFloat4::load(50.0, 55.0, 60.0, 65.0));
            let joint_weights = [SimdFloat4::load(1.0, -1.0, 2.0, 0.1)];

            let mut layers = [Layer::new(), Layer::new()];
            layers[0].weight = 2.0;
            layers[0].transform = &input_transforms[0];
            layers[1].weight = 3.0;
            layers[1].transform = &input_transforms[1];
            layers[1].joint_weights = &joint_weights;

            let mut output_transforms = [SoaTransform::identity(); 1];

            let mut job = BlendingJob::new();
            job.layers = &layers;
            job.bind_pose = &bind_poses;
            job.output = &mut output_transforms;

            assert_eq!(job.run(), true);

            expect_soa_float3_eq!(output_transforms[0].translation, 8.0, 10.0,
                                150.0 / 8.0, 47.5 / 2.3, 28.0, 30.0, 310.0 / 8.0,
                                93.5 / 2.3, 48.0, 50.0, 470.0 / 8.0, 139.5 / 2.3);
            expect_soa_float3_eq!(output_transforms[0].scale, 1.0, 1.0, 1.0, 1.0, 1.0,
                                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        }
    }

    #[test]
    fn threshold() {
        let identity = SoaTransform::identity();

        // Initialize inputs.
        let mut input_transforms = [[identity], [identity]];

        // Initialize bind pose.
        let mut bind_poses = [identity];
        bind_poses[0].scale = SoaFloat3::load(
            SimdFloat4::load(0.0, 1.0, 2.0, 3.0),
            SimdFloat4::load(4.0, 5.0, 6.0, 7.0),
            SimdFloat4::load(8.0, 9.0, 10.0, 11.0));

        input_transforms[0][0].translation = SoaFloat3::load(
            SimdFloat4::load(2.0, 3.0, 4.0, 5.0),
            SimdFloat4::load(6.0, 7.0, 8.0, 9.0),
            SimdFloat4::load(10.0, 11.0, 12.0, 13.0));
        input_transforms[1][0].translation = SoaFloat3::load(
            SimdFloat4::load(3.0, 4.0, 5.0, 6.0),
            SimdFloat4::load(7.0, 8.0, 9.0, 10.0),
            SimdFloat4::load(11.0, 12.0, 13.0, 14.0));

        {  // Threshold is not reached.
            let mut layers = [Layer::new(), Layer::new()];
            layers[0].weight = 0.04;
            layers[0].transform = &input_transforms[0];
            layers[1].weight = 0.06;
            layers[1].transform = &input_transforms[1];

            let mut output_transforms = [SoaTransform::identity(); 1];

            let mut job = BlendingJob::new();
            job.threshold = 0.1;
            job.layers = &layers;
            job.bind_pose = &bind_poses;
            job.output = &mut output_transforms;

            assert_eq!(job.run(), true);

            expect_soa_float3_eq!(output_transforms[0].translation, 2.6, 3.6, 4.6,
                                5.6, 6.6, 7.6, 8.6, 9.6, 10.6, 11.6, 12.6,
                                13.6);
            expect_soa_quaternion_eq_est!(output_transforms[0].rotation, 0.0, 0.0, 0.0,
                                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                        1.0, 1.0, 1.0, 1.0);
            expect_soa_float3_eq!(output_transforms[0].scale, 1.0, 1.0, 1.0, 1.0, 1.0,
                                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        }
        {  // Threshold is reached at 100%.
            let mut layers = [Layer::new(), Layer::new()];
            layers[0].weight = 1e-27;
            layers[0].transform = &input_transforms[0];
            layers[1].weight = 0.0;
            layers[1].transform = &input_transforms[1];

            let mut output_transforms = [SoaTransform::identity(); 1];

            let mut job = BlendingJob::new();
            job.threshold = 0.1;
            job.layers = &layers;
            job.bind_pose = &bind_poses;
            job.output = &mut output_transforms;

            assert_eq!(job.run(), true);

            expect_soa_float3_eq!(output_transforms[0].translation, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            expect_soa_quaternion_eq_est!(output_transforms[0].rotation, 0.0, 0.0, 0.0,
                                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                        1.0, 1.0, 1.0, 1.0);
            expect_soa_float3_eq!(output_transforms[0].scale, 0.0, 1.0, 2.0, 3.0, 4.0,
                                5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0);
        }
    }

    #[test]
    fn additive_weight() {
        let identity = SoaTransform::identity();

        // Initialize inputs.
        let mut input_transforms = [[identity], [identity]];
        input_transforms[0][0].translation = SoaFloat3::load(
            SimdFloat4::load(0.0, 1.0, 2.0, 3.0),
            SimdFloat4::load(4.0, 5.0, 6.0, 7.0),
            SimdFloat4::load(8.0, 9.0, 10.0, 11.0));
        input_transforms[0][0].rotation = SoaQuaternion::load(
            SimdFloat4::load(0.70710677, 0.0, 0.0, 0.382683432),
            SimdFloat4::load(0.0, 0.0, 0.70710677, 0.0),
            SimdFloat4::load(0.0, 0.0, 0.0, 0.0),
            SimdFloat4::load(0.70710677, 1.0, -0.70710677, 0.9238795));
        input_transforms[0][0].scale = SoaFloat3::load(
            SimdFloat4::load(12.0, 13.0, 14.0, 15.0),
            SimdFloat4::load(16.0, 17.0, 18.0, 19.0),
            SimdFloat4::load(20.0, 21.0, 22.0, 23.0));
        input_transforms[1][0].translation = -input_transforms[0][0].translation;
        input_transforms[1][0].rotation = input_transforms[0][0].rotation.conjugate();
        input_transforms[1][0].scale = -input_transforms[0][0].scale;

        // Initialize bind pose.
        let bind_poses = [identity];

        {
            let mut layers = [Layer::new()];
            layers[0].transform = &input_transforms[0];

            // No weight for the 1st layer.
            layers[0].weight = 0.0;

            let mut output_transforms = [SoaTransform::identity()];

            let mut job = BlendingJob::new();
            job.additive_layers = &layers;
            job.bind_pose = &bind_poses;
            job.output = &mut output_transforms;
            assert_eq!(job.run(), true);

            expect_soa_float3_eq!(output_transforms[0].translation, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            expect_soa_quaternion_eq_est!(output_transforms[0].rotation, 0.0, 0.0, 0.0,
                                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                        1.0, 1.0, 1.0, 1.0);
            expect_soa_float3_eq!(output_transforms[0].scale, 1.0, 1.0, 1.0, 1.0, 1.0,
                                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        }

        {
            let mut layers = [Layer::new()];
            layers[0].transform = &input_transforms[0];

            let mut output_transforms = [SoaTransform::identity()];

            // .5 weight for the 1st layer.
            layers[0].weight = 0.5;

            let mut job = BlendingJob::new();
            job.additive_layers = &layers;
            job.bind_pose = &bind_poses;
            job.output = &mut output_transforms;
            assert_eq!(job.run(), true);

            expect_soa_float3_eq!(output_transforms[0].translation, 0.0, 0.5, 1.0, 1.5,
                                2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5);
            expect_soa_quaternion_eq_est!(output_transforms[0].rotation, 0.3826834, 0.0,
                                        0.0, 0.19509032, 0.0, 0.0, -0.3826834, 0.0, 0.0,
                                        0.0, 0.0, 0.0, 0.9238795, 1.0, 0.9238795,
                                        0.98078528);
            expect_soa_float3_eq!(output_transforms[0].scale, 6.5, 7.0, 7.5, 8.0, 8.5,
                                9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0);
        }

        {
            let mut layers = [Layer::new()];
            layers[0].transform = &input_transforms[0];

            let mut output_transforms = [SoaTransform::identity()];

            // Full weight for the 1st layer.
            layers[0].weight = 1.0;

            let mut job = BlendingJob::new();
            job.additive_layers = &layers;
            job.bind_pose = &bind_poses;
            job.output = &mut output_transforms;
            assert_eq!(job.run(), true);

            expect_soa_float3_eq!(output_transforms[0].translation, 0.0, 1.0, 2.0, 3.0,
                                4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0);
            expect_soa_quaternion_eq_est!(output_transforms[0].rotation, 0.70710677, 0.0,
                                        0.0, 0.382683432, 0.0, 0.0, -0.70710677, 0.0,
                                        0.0, 0.0, 0.0, 0.0, 0.70710677, 1.0, 0.70710677,
                                        0.9238795);
            expect_soa_float3_eq!(output_transforms[0].scale, 12.0, 13.0, 14.0, 15.0,
                                16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0);
        }

        {
            let mut layers = [Layer::new(), Layer::new()];
            layers[0].transform = &input_transforms[0];
            layers[1].transform = &input_transforms[1];

            let mut output_transforms = [SoaTransform::identity(); 1];

            // No weight for the 1st layer.
            layers[0].weight = 0.0;
            layers[1].weight = 1.0;

            let mut job = BlendingJob::new();
            job.additive_layers = &layers;
            job.bind_pose = &bind_poses;
            job.output = &mut output_transforms;
            assert_eq!(job.run(), true);

            expect_soa_float3_eq!(output_transforms[0].translation, -0.0, -1.0, -2.0,
                                -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0, -11.0);
            expect_soa_quaternion_eq_est!(output_transforms[0].rotation, -0.70710677,
                                        -0.0, -0.0, -0.382683432, -0.0, -0.0,
                                        0.70710677, -0.0, -0.0, -0.0, -0.0, -0.0,
                                        0.70710677, 1.0, 0.70710677, 0.9238795);
            expect_soa_float3_eq!(output_transforms[0].scale, -12.0, -13.0, -14.0, -15.0,
                                -16.0, -17.0, -18.0, -19.0, -20.0, -21.0, -22.0, -23.0);
        }
        {
            let mut layers = [Layer::new(), Layer::new()];
            layers[0].transform = &input_transforms[0];
            layers[1].transform = &input_transforms[1];

            let mut output_transforms = [SoaTransform::identity(); 1];

            // Full weight for the both layer.
            layers[0].weight = 1.0;
            layers[1].weight = 1.0;

            let mut job = BlendingJob::new();
            job.additive_layers = &layers;
            job.bind_pose = &bind_poses;
            job.output = &mut output_transforms;
            assert_eq!(job.run(), true);

            expect_soa_float3_eq!(output_transforms[0].translation, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            expect_soa_quaternion_eq_est!(output_transforms[0].rotation, 0.0, 0.0, 0.0,
                                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                        1.0, 1.0, 1.0, 1.0);
            expect_soa_float3_eq!(output_transforms[0].scale, -144.0, -169.0, -196.0,
                                -225.0, -256.0, -289.0, -324.0, -361.0, -400.0, -441.0,
                                -484.0, -529.0);
        }
        {
            let mut layers = [Layer::new(), Layer::new()];
            layers[0].transform = &input_transforms[0];
            layers[1].transform = &input_transforms[1];

            let mut output_transforms = [SoaTransform::identity(); 1];

            // Subtract second layer.
            layers[0].weight = 0.5;
            layers[1].transform = &input_transforms[0];
            layers[1].weight = -0.5;

            let mut job = BlendingJob::new();
            job.additive_layers = &layers;
            job.bind_pose = &bind_poses;
            job.output = &mut output_transforms;
            assert_eq!(job.run(), true);

            expect_soa_float3_eq!(output_transforms[0].translation, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            expect_soa_quaternion_eq_est!(output_transforms[0].rotation, 0.0, 0.0, 0.0,
                                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                        1.0, 1.0, 1.0, 1.0);
            expect_soa_float3_eq!(output_transforms[0].scale, 1.0, 1.0, 1.0, 1.0, 1.0,
                                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        }
    }

    #[test]
    fn additive_joint_weight() {
        let identity = SoaTransform::identity();

        // Initialize inputs.
        let mut input_transforms = [identity];
        input_transforms[0].translation = SoaFloat3::load(
            SimdFloat4::load(0.0, 1.0, 2.0, 3.0),
            SimdFloat4::load(4.0, 5.0, 6.0, 7.0),
            SimdFloat4::load(8.0, 9.0, 10.0, 11.0));
        input_transforms[0].rotation = SoaQuaternion::load(
            SimdFloat4::load(0.70710677, 0.0, 0.0, 0.382683432),
            SimdFloat4::load(0.0, 0.0, 0.70710677, 0.0),
            SimdFloat4::load(0.0, 0.0, 0.0, 0.0),
            SimdFloat4::load(0.70710677, 1.0, -0.70710677, 0.9238795));
        input_transforms[0].scale = SoaFloat3::load(
            SimdFloat4::load(12.0, 13.0, 14.0, 15.0),
            SimdFloat4::load(16.0, 17.0, 18.0, 19.0),
            SimdFloat4::load(20.0, 21.0, 22.0, 23.0));

        let joint_weights = [
            SimdFloat4::load(1.0, 0.5, 0.0, -1.0)];

        // Initialize bind pose.
        let bind_poses = [identity];

        {
            let mut layers = [Layer::new()];
            layers[0].transform = &input_transforms;
            layers[0].joint_weights = &joint_weights;

            let mut output_transforms = [SoaTransform::identity()];

            // No weight for the 1st layer.
            layers[0].weight = 0.0;

            let mut job = BlendingJob::new();
            job.additive_layers = &layers;
            job.bind_pose = &bind_poses;
            job.output = &mut output_transforms;
            assert_eq!(job.run(), true);

            expect_soa_float3_eq!(output_transforms[0].translation, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            expect_soa_quaternion_eq_est!(output_transforms[0].rotation, 0.0, 0.0, 0.0,
                                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                        1.0, 1.0, 1.0, 1.0);
            expect_soa_float3_eq!(output_transforms[0].scale, 1.0, 1.0, 1.0, 1.0, 1.0,
                                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        }
        {
            let mut layers = [Layer::new()];
            layers[0].transform = &input_transforms;
            layers[0].joint_weights = &joint_weights;

            let mut output_transforms = [SoaTransform::identity()];

            // .5 weight for the 1st layer.
            layers[0].weight = 0.5;

            let mut job = BlendingJob::new();
            job.additive_layers = &layers;
            job.bind_pose = &bind_poses;
            job.output = &mut output_transforms;
            assert_eq!(job.run(), true);

            expect_soa_float3_eq!(output_transforms[0].translation, 0.0, 0.25, 0.0, 0.0,
                                2.0, 1.25, 0.0, 0.0, 4.0, 2.25, 0.0, 0.0);
            expect_soa_quaternion_eq_est!(output_transforms[0].rotation, 0.3826834, 0.0,
                                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                        0.0, 0.9238795, 1.0, 1.0, 1.0);
            expect_soa_float3_eq!(output_transforms[0].scale, 6.5, 4.0, 1.0, 1.0, 8.5,
                                5.0, 1.0, 1.0, 10.5, 6.0, 1.0, 1.0);
        }
        {
            let mut layers = [Layer::new()];
            layers[0].transform = &input_transforms;
            layers[0].joint_weights = &joint_weights;

            let mut output_transforms = [SoaTransform::identity()];

            // Full weight for the 1st layer.
            layers[0].weight = 1.0;

            let mut job = BlendingJob::new();
            job.additive_layers = &layers;
            job.bind_pose = &bind_poses;
            job.output = &mut output_transforms;
            assert_eq!(job.run(), true);

            expect_soa_float3_eq!(output_transforms[0].translation, 0.0, 0.5, 0.0, 0.0,
                                4.0, 2.5, 0.0, 0.0, 8.0, 4.5, 0.0, 0.0);
            expect_soa_quaternion_eq_est!(output_transforms[0].rotation, 0.70710677, 0.0,
                                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                        0.0, 0.70710677, 1.0, 1.0, 1.0);
            expect_soa_float3_eq!(output_transforms[0].scale, 12.0, 7.0, 1.0, 1.0, 16.0,
                                9.0, 1.0, 1.0, 20.0, 11.0, 1.0, 1.0);
        }
        {
            let mut layers = [Layer::new()];
            layers[0].transform = &input_transforms;
            layers[0].joint_weights = &joint_weights;

            let mut output_transforms = [SoaTransform::identity()];

            // Subtract layer.
            layers[0].weight = -1.0;

            let mut job = BlendingJob::new();
            job.additive_layers = &layers;
            job.bind_pose = &bind_poses;
            job.output = &mut output_transforms;
            assert_eq!(job.run(), true);

            expect_soa_float3_eq!(output_transforms[0].translation, -0.0, -0.5, 0.0, 0.0,
                                -4.0, -2.5, 0.0, 0.0, -8.0, -4.5, 0.0, 0.0);
            expect_soa_quaternion_eq_est!(output_transforms[0].rotation, -0.70710677, 0.0,
                                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                        0.0, 0.70710677, 1.0, 1.0, 1.0);
            expect_soa_float3_eq_est!(output_transforms[0].scale, 1.0 / 12.0, 1.0 / 7.0,
                                    1.0, 1.0, 1.0 / 16.0, 1.0 / 9.0, 1.0, 1.0,
                                    1.0 / 20.0, 1.0 / 11.0, 1.0, 1.0);
        }
    }
}