/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use crate::soa_transform::SoaTransform;
use crate::simd_math::SimdFloat4;

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
    pub output: &'a [SoaTransform],
}

impl<'a> BlendingJob<'a> {
    // Default constructor, initializes default values.
    pub fn new() -> BlendingJob<'a> {
        return BlendingJob {
            threshold: 0.1,
            layers: &[],
            additive_layers: &[],
            bind_pose: &[],
            output: &[],
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
    pub fn run(&mut self) -> bool {
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
    job: &'a BlendingJob<'a>,

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
    fn new(_job: &'a BlendingJob<'a>) -> ProcessArgs<'a> {
        let args = ProcessArgs {
            accumulated_weights: [SimdFloat4::zero(); crate::skeleton::Constants::KMaxSoAJoints as usize],
            job: _job,
            num_soa_joints: _job.bind_pose.len(),
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
fn blend_layers(_args: &mut ProcessArgs) {
    todo!()
}

// Blends bind pose to the output if accumulated weight is less than the
// threshold value.
fn blend_bind_pose(_args: &mut ProcessArgs) {
    todo!()
}

// Normalizes output rotations. Quaternion length cannot be zero as opposed
// quaternions have been fixed up during blending passes.
// Translations and scales are already normalized because weights were
// pre-multiplied by the normalization ratio.
fn normalize(_args: &mut ProcessArgs) {
    todo!()
}

// Process additive blending pass.
fn add_layers(_args: &mut ProcessArgs) {
    todo!()
}