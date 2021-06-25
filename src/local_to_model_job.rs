/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use crate::skeleton::Skeleton;
use crate::simd_math::{Float4x4, SimdFloat4};
use crate::soa_transform::SoaTransform;
use crate::soa_float4x4::SoaFloat4x4;

// Computes model-space joint matrices from local-space SoaTransform.
// This job uses the skeleton to define joints parent-child hierarchy. The job
// iterates through all joints to compute their transform relatively to the
// skeleton root.
// Job inputs is an array of SoaTransform objects (in local-space), ordered like
// skeleton's joints. Job output is an array of matrices (in model-space),
// ordered like skeleton's joints. Output are matrices, because the combination
// of affine transformations can contain shearing or complex transformation
// that cannot be represented as Transform object.
pub struct LocalToModelJob<'a> {
    // Job input.

    // The Skeleton object describing the joint hierarchy used for local to
    // model space conversion.
    pub skeleton: Option<&'a Skeleton>,

    // The root matrix will multiply to every model space matrices, default nullptr
    // means an identity matrix. This can be used to directly compute world-space
    // transforms for example.
    pub root: Option<&'a Float4x4>,

    // Defines "from" which joint the local-to-model conversion should start.
    // Default value is ozz::Skeleton::kNoParent, meaning the whole hierarchy is
    // updated. This parameter can be used to optimize update by limiting
    // conversion to part of the joint hierarchy. Note that "from" parent should
    // be a valid matrix, as it is going to be used as part of "from" joint
    // hierarchy update.
    pub from: i32,

    // Defines "to" which joint the local-to-model conversion should go, "to"
    // included. Update will end before "to" joint is reached if "to" is not part
    // of the hierarchy starting from "from". Default value is
    // ozz::animation::Skeleton::kMaxJoints, meaning the hierarchy (starting from
    // "from") is updated to the last joint.
    pub to: i32,

    // If true, "from" joint is not updated during job execution. Update starts
    // with all children of "from". This can be used to update a model-space
    // transform independently from the local-space one. To do so: set "from"
    // joint model-space transform matrix, and run this Job with "from_excluded"
    // to update all "from" children.
    // Default value is false.
    pub from_excluded: bool,

    // The input range that store local transforms.
    pub input: &'a [SoaTransform],

    // Job output.

    // The output range to be filled with model-space matrices.
    pub output: &'a mut [Float4x4],
}

impl<'a> LocalToModelJob<'a> {
    // Default constructor, initializes default values.
    pub fn new() -> LocalToModelJob<'a> {
        return LocalToModelJob {
            skeleton: None,
            root: None,
            from: crate::skeleton::Constants::KNoParent as i32,
            to: crate::skeleton::Constants::KMaxJoints as i32,
            from_excluded: false,
            input: &[],
            output: &mut [],
        };
    }

    // Validates job parameters. Returns true for a valid job, or false otherwise:
    // -if any input pointer, including ranges, is nullptr.
    // -if the size of the input is smaller than the skeleton's number of joints.
    // Note that this input has a SoA format.
    // -if the size of of the output is smaller than the skeleton's number of
    // joints.
    pub fn validate(&self) -> bool {
        // Don't need any early out, as jobs are valid in most of the performance
        // critical cases.
        // Tests are written in multiple lines in order to avoid branches.
        let mut valid = true;

        // Test for nullptr begin pointers.
        if self.skeleton.is_none() {
            return false;
        }

        let num_joints = self.skeleton.unwrap().num_joints() as usize;
        let num_soa_joints = (num_joints + 3) / 4;

        // Test input and output ranges, implicitly tests for nullptr end pointers.
        valid &= self.input.len() >= num_soa_joints;
        valid &= self.output.len() >= num_joints;

        return valid;
    }

    // Runs job's local-to-model task.
    // The job is validated before any operation is performed, see validate() for
    // more details.
    // Returns false if job is not valid. See validate() function.
    pub fn run(&'a mut self) -> bool {
        if !self.validate() {
            return false;
        }

        let parents = self.skeleton.unwrap().joint_parents();

        // Initializes an identity matrix that will be used to compute roots model
        // matrices without requiring a branch.
        let identity = Float4x4::identity();
        let root_matrix = match self.root.is_none() {
            true => &identity,
            false => self.root.unwrap()
        };

        // Applies hierarchical transformation.
        // Loop ends after "to".
        let end = i32::min(self.to + 1, self.skeleton.unwrap().num_joints());
        // Begins iteration from "from", or the next joint if "from" is excluded.
        // Process next joint if end is not reach. parents[begin] >= from is true as
        // long as "begin" is a child of "from".
        let mut i = i32::max(self.from + self.from_excluded as i32, 0);
        let mut process = i < end && (!self.from_excluded || parents[i as usize] >= self.from as i16);
        while process {
            // Builds soa matrices from soa transforms.
            let transform = &self.input[i as usize / 4];
            let local_soa_matrices = SoaFloat4x4::from_affine(
                &transform.translation, &transform.rotation, &transform.scale);

            // Converts to aos matrices.
            let mut local_aos_matrices = [Float4x4::identity(); 4];
            SimdFloat4::transpose16x16(&local_soa_matrices.cols,
                                       &mut local_aos_matrices);

            // parents[i] >= from is true as long as "i" is a child of "from".
            let soa_end = (i + 4) & !3;
            while i < soa_end && process {
                let parent = parents[i as usize];
                let parent_matrix = match parent == crate::skeleton::Constants::KNoParent as i16 {
                    true => root_matrix,
                    false => &self.output[parent as usize]
                };
                self.output[i as usize] = *parent_matrix * local_aos_matrices[i as usize & 3];

                i += 1;
                process = i < end && parents[i as usize] >= self.from as i16;
            }
        }
        return true;
    }
}