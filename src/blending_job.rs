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
        todo!()
    }

    // Runs job's blending task.
    // The job is validated before any operation is performed, see validate() for
    // more details.
    // Returns false if *this job is not valid.
    pub fn run(&mut self) -> bool {
        todo!()
    }
}