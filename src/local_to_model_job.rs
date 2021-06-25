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

//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------
#[cfg(test)]
mod local_to_model {
    use crate::raw_skeleton::{RawSkeleton, Joint};
    use crate::skeleton_builder::SkeletonBuilder;
    use crate::soa_transform::SoaTransform;
    use crate::simd_math::*;
    use crate::local_to_model_job::LocalToModelJob;
    use crate::soa_float::SoaFloat3;
    use crate::soa_quaternion::SoaQuaternion;
    use crate::math_test_helper::*;
    use crate::*;
    use crate::skeleton::Skeleton;

    #[test]
    fn job_validity() {
        let mut raw_skeleton = RawSkeleton::new();

        // empty skeleton.
        let empty_skeleton = SkeletonBuilder::apply(&raw_skeleton);
        assert_eq!(empty_skeleton.is_some(), true);

        // Adds 2 joints.
        raw_skeleton.roots.resize(1, Joint::new());
        let root = &mut raw_skeleton.roots[0];
        root.name = "root".to_string();
        root.children.resize(1, Joint::new());

        let skeleton = SkeletonBuilder::apply(&raw_skeleton);
        assert_eq!(skeleton.is_some(), true);

        let input = [SoaTransform::identity(), SoaTransform::identity()];
        let mut output = [Float4x4::identity(); 5];

        // Default job
        {
            let mut job = LocalToModelJob::new();
            assert_eq!(job.validate(), false);
            assert_eq!(job.run(), false);
        }

        // nullptr output
        {
            let mut job = LocalToModelJob::new();
            job.skeleton = skeleton.as_ref();
            job.input = &input[0..1];
            assert_eq!(job.validate(), false);
            assert_eq!(job.run(), false);
        }
        // nullptr input
        {
            let mut job = LocalToModelJob::new();
            job.skeleton = skeleton.as_ref();
            job.output = &mut output[0..2];
            assert_eq!(job.validate(), false);
            assert_eq!(job.run(), false);
        }
        // Null skeleton.
        {
            let mut job = LocalToModelJob::new();
            job.input = &input[0..1];
            job.output = &mut output[0..4];
            assert_eq!(job.validate(), false);
            assert_eq!(job.run(), false);
        }
        // Invalid output range: end < begin.
        {
            let mut job = LocalToModelJob::new();
            job.skeleton = skeleton.as_ref();
            job.input = &input[0..1];
            job.output = &mut output[0..1];
            assert_eq!(job.validate(), false);
            assert_eq!(job.run(), false);
        }
        // Invalid output range: too small.
        {
            let mut job = LocalToModelJob::new();
            job.skeleton = skeleton.as_ref();
            job.input = &input[0..1];
            job.output = &mut output[0..1];
            assert_eq!(job.validate(), false);
            assert_eq!(job.run(), false);
        }
        // Invalid input range: too small.
        {
            let mut job = LocalToModelJob::new();
            job.skeleton = skeleton.as_ref();
            job.input = &input[0..0];
            job.output = &mut output;
            assert_eq!(job.validate(), false);
            assert_eq!(job.run(), false);
        }
        // Valid job.
        {
            let mut job = LocalToModelJob::new();
            job.skeleton = skeleton.as_ref();
            job.input = &input;
            job.output = &mut output[0..2];
            assert_eq!(job.validate(), true);
            assert_eq!(job.run(), true);
        }
        // Valid job with root matrix.
        {
            let mut job = LocalToModelJob::new();
            let v = SimdFloat4::load(4.0, 3.0, 2.0, 1.0);
            let world = Float4x4::translation(v);
            job.skeleton = skeleton.as_ref();
            job.root = Some(&world);
            job.input = &input;
            job.output = &mut output[0..2];
            assert_eq!(job.validate(), true);
            assert_eq!(job.run(), true);
        }
        // Valid out-of-bound from.
        {
            let mut job = LocalToModelJob::new();
            job.skeleton = skeleton.as_ref();
            job.from = 93;
            job.input = &input;
            job.output = &mut output[0..2];
            assert_eq!(job.validate(), true);
            assert_eq!(job.run(), true);
        }
        // Valid out-of-bound from.
        {
            let mut job = LocalToModelJob::new();
            job.skeleton = skeleton.as_ref();
            job.from = -93;
            job.input = &input;
            job.output = &mut output[0..2];
            assert_eq!(job.validate(), true);
            assert_eq!(job.run(), true);
        }
        // Valid out-of-bound to.
        {
            let mut job = LocalToModelJob::new();
            job.skeleton = skeleton.as_ref();
            job.from = 93;
            job.input = &input;
            job.output = &mut output[0..2];
            assert_eq!(job.validate(), true);
            assert_eq!(job.run(), true);
        }
        // Valid out-of-bound to.
        {
            let mut job = LocalToModelJob::new();
            job.skeleton = skeleton.as_ref();
            job.from = -93;
            job.input = &input;
            job.output = &mut output[0..2];
            assert_eq!(job.validate(), true);
            assert_eq!(job.run(), true);
        }
        // Valid job with empty skeleton.
        {
            let mut job = LocalToModelJob::new();
            job.skeleton = empty_skeleton.as_ref();
            job.input = &input[0..0];
            job.output = &mut output[0..0];
            assert_eq!(job.validate(), true);
            assert_eq!(job.run(), true);
        }
        // Valid job. Bigger input & output
        {
            let mut job = LocalToModelJob::new();
            job.skeleton = skeleton.as_ref();
            job.input = &input;
            job.output = &mut output;
            assert_eq!(job.validate(), true);
            assert_eq!(job.run(), true);
        }
    }

    #[test]
    fn transformation() {
        // Builds the skeleton
        /*
         6 joints
           j0
          /  \
         j1  j3
          |  / \
         j2 j4 j5
        */
        let mut raw_skeleton = RawSkeleton::new();
        raw_skeleton.roots.resize(1, Joint::new());
        let root = &mut raw_skeleton.roots[0];
        root.name = "j0".to_string();

        root.children.resize(2, Joint::new());
        root.children[0].name = "j1".to_string();
        root.children[1].name = "j3".to_string();

        root.children[0].children.resize(1, Joint::new());
        root.children[0].children[0].name = "j2".to_string();

        root.children[1].children.resize(2, Joint::new());
        root.children[1].children[0].name = "j4".to_string();
        root.children[1].children[1].name = "j5".to_string();

        assert_eq!(raw_skeleton.validate(), true);
        assert_eq!(raw_skeleton.num_joints(), 6);

        let skeleton = SkeletonBuilder::apply(&raw_skeleton);
        assert_eq!(skeleton.is_some(), true);

        // Initializes an input transformation.
        let input = [
            // Stores up to 8 inputs, needs 6.
            SoaTransform {
                translation: SoaFloat3::load(SimdFloat4::load(2.0, 0.0, 1.0, -2.0),
                                             SimdFloat4::load(2.0, 0.0, 2.0, -2.0),
                                             SimdFloat4::load(2.0, 0.0, 4.0, -2.0)),
                rotation: SoaQuaternion::load(SimdFloat4::load(0.0, 0.0, 0.0, 0.0),
                                              SimdFloat4::load(0.0, 0.70710677, 0.0, 0.0),
                                              SimdFloat4::load(0.0, 0.0, 0.0, 0.0),
                                              SimdFloat4::load(1.0, 0.70710677, 1.0, 1.0)),
                scale: SoaFloat3::load(SimdFloat4::load(1.0, 1.0, 1.0, 10.0),
                                       SimdFloat4::load(1.0, 1.0, 1.0, 10.0),
                                       SimdFloat4::load(1.0, 1.0, 1.0, 10.0)),
            },
            SoaTransform {
                translation: SoaFloat3::load(SimdFloat4::load(12.0, 0.0, 0.0, 0.0),
                                             SimdFloat4::load(46.0, 0.0, 0.0, 0.0),
                                             SimdFloat4::load(-12.0, 0.0, 0.0, 0.0)),
                rotation: SoaQuaternion::load(SimdFloat4::load(0.0, 0.0, 0.0, 0.0),
                                              SimdFloat4::load(0.0, 0.0, 0.0, 0.0),
                                              SimdFloat4::load(0.0, 0.0, 0.0, 0.0),
                                              SimdFloat4::load(1.0, 1.0, 1.0, 1.0)),
                scale: SoaFloat3::load(SimdFloat4::load(1.0, -0.1, 1.0, 1.0),
                                       SimdFloat4::load(1.0, -0.1, 1.0, 1.0),
                                       SimdFloat4::load(1.0, -0.1, 1.0, 1.0)),
            }];

        {
            // Prepares the job with root == nullptr (default identity matrix)
            let mut output = [Float4x4::identity(); 6];
            let mut job = LocalToModelJob::new();
            job.skeleton = skeleton.as_ref();
            job.input = &input;
            job.output = &mut output;
            assert_eq!(job.validate(), true);
            assert_eq!(job.run(), true);
            expect_float4x4_eq!(output[0], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 2.0, 2.0, 2.0, 1.0);
            expect_float4x4_eq!(output[1], 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
                               0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 1.0);
            expect_float4x4_eq!(output[2], 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
                               0.0, 0.0, 0.0, 6.0, 4.0, 1.0, 1.0);
            expect_float4x4_eq!(output[3], 10.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0,
                               0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[4], 10.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0,
                               0.0, 10.0, 0.0, 120.0, 460.0, -120.0, 1.0);
            expect_float4x4_eq!(output[5], -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0,
                               0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        }

        {
            // Prepares the job with root == Translation(4,3,2,1)
            let mut output = [Float4x4::identity(); 6];
            let v = SimdFloat4::load(4.0, 3.0, 2.0, 1.0);
            let world = Float4x4::translation(v);
            let mut job = LocalToModelJob::new();
            job.skeleton = skeleton.as_ref();
            job.root = Some(&world);
            job.input = &input;
            job.output = &mut output;
            assert_eq!(job.validate(), true);
            assert_eq!(job.run(), true);
            expect_float4x4_eq!(output[0], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 6.0, 5.0, 4.0, 1.0);
            expect_float4x4_eq!(output[1], 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
                               0.0, 0.0, 0.0, 6.0, 5.0, 4.0, 1.0);
            expect_float4x4_eq!(output[2], 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
                               0.0, 0.0, 0.0, 10.0, 7.0, 3.0, 1.0);
            expect_float4x4_eq!(output[3], 10.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0,
                               0.0, 10.0, 0.0, 4.0, 3.0, 2.0, 1.0);
            expect_float4x4_eq!(output[4], 10.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0,
                               0.0, 10.0, 0.0, 124.0, 463.0, -118.0, 1.0);
            expect_float4x4_eq!(output[5], -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0,
                               0.0, -1.0, 0.0, 4.0, 3.0, 2.0, 1.0);
        }
    }

    #[test]
    fn transformation_from_to() {
        // Builds the skeleton
        /*
         6 joints
               *
             /   \
           j0    j7
          /  \
         j1  j3
          |  / \
         j2 j4 j6
             |
            j5
        */
        let mut raw_skeleton = RawSkeleton::new();
        raw_skeleton.roots.resize(2, Joint::new());
        let j0 = &mut raw_skeleton.roots[0];
        j0.name = "j0".to_string();
        let j7 = &mut raw_skeleton.roots[1];
        j7.name = "j7".to_string();

        j0.children.resize(2, Joint::new());
        j0.children[0].name = "j1".to_string();
        j0.children[1].name = "j3".to_string();

        j0.children[0].children.resize(1, Joint::new());
        j0.children[0].children[0].name = "j2".to_string();

        j0.children[1].children.resize(2, Joint::new());
        j0.children[1].children[0].name = "j4".to_string();
        j0.children[1].children[1].name = "j6".to_string();

        j0.children[1].children[0].children.resize(1, Joint::new());
        j0.children[1].children[0].children[0].name = "j5".to_string();

        assert_eq!(raw_skeleton.validate(), true);
        assert_eq!(raw_skeleton.num_joints(), 8);

        let skeleton = SkeletonBuilder::apply(&raw_skeleton);
        assert_eq!(skeleton.is_some(), true);

        // Initializes an input transformation.
        let input = [
            // Stores up to 8 inputs, needs 7.
            //                             j0   j1   j2    j3
            SoaTransform {
                translation: SoaFloat3::load(SimdFloat4::load(2.0, 0.0, -2.0, 1.0),
                                             SimdFloat4::load(2.0, 0.0, -2.0, 2.0),
                                             SimdFloat4::load(2.0, 0.0, -2.0, 4.0)),
                rotation: SoaQuaternion::load(SimdFloat4::load(0.0, 0.0, 0.0, 0.0),
                                              SimdFloat4::load(0.0, 0.70710677, 0.0, 0.0),
                                              SimdFloat4::load(0.0, 0.0, 0.0, 0.0),
                                              SimdFloat4::load(1.0, 0.70710677, 1.0, 1.0)),
                scale: SoaFloat3::load(SimdFloat4::load(1.0, 1.0, 10.0, 1.0),
                                       SimdFloat4::load(1.0, 1.0, 10.0, 1.0),
                                       SimdFloat4::load(1.0, 1.0, 10.0, 1.0)),
            },
            //                             j4    j5   j6   j7.
            SoaTransform {
                translation: SoaFloat3::load(SimdFloat4::load(12.0, 0.0, 3.0, 6.0),
                                             SimdFloat4::load(46.0, 0.0, 4.0, 7.0),
                                             SimdFloat4::load(-12.0, 0.0, 5.0, 8.0)),
                rotation: SoaQuaternion::load(SimdFloat4::load(0.0, 0.0, 0.0, 0.0),
                                              SimdFloat4::load(0.0, 0.0, 0.0, 0.0),
                                              SimdFloat4::load(0.0, 0.0, 0.0, 0.0),
                                              SimdFloat4::load(1.0, 1.0, 1.0, 1.0)),
                scale: SoaFloat3::load(SimdFloat4::load(1.0, -0.1, 1.0, 1.0),
                                       SimdFloat4::load(1.0, -0.1, 1.0, 1.0),
                                       SimdFloat4::load(1.0, -0.1, 1.0, 1.0)),
            }];

        let mut output = [Float4x4::identity(); 8];
        let mut job_full = LocalToModelJob::new();
        {  // Intialize whole hierarchy output
            job_full.skeleton = skeleton.as_ref();
            job_full.from = crate::skeleton::Constants::KNoParent as i32;
            job_full.input = &input;
            job_full.output = &mut output;
            assert_eq!(job_full.validate(), true);
            assert_eq!(job_full.run(), true);
            expect_float4x4_eq!(output[0], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 2.0, 2.0, 2.0, 1.0);
            expect_float4x4_eq!(output[1], 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
                               0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 1.0);
            expect_float4x4_eq!(output[2], 0.0, 0.0, -10.0, 0.0, 0.0, 10.0, 0.0, 0.0,
                               10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 1.0);
            expect_float4x4_eq!(output[3], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 3.0, 4.0, 6.0, 1.0);
            expect_float4x4_eq!(output[4], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 15.0, 50.0, -6.0, 1.0);
            expect_float4x4_eq!(output[5], -0.1, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0,
                               0.0, -0.1, 0.0, 15.0, 50.0, -6.0, 1.0);
            expect_float4x4_eq!(output[6], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 6.0, 8.0, 11.0, 1.0);
            expect_float4x4_eq!(output[7], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 6.0, 7.0, 8.0, 1.0);
        }

        {  // Updates from j0, j7 shouldn't be updated
            output.fill(Float4x4::identity());

            let mut job = LocalToModelJob::new();
            job.skeleton = skeleton.as_ref();
            job.from = 0;
            job.input = &input;
            job.output = &mut output;
            assert_eq!(job.validate(), true);
            assert_eq!(job.run(), true);

            expect_float4x4_eq!(output[0], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 2.0, 2.0, 2.0, 1.0);
            expect_float4x4_eq!(output[1], 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
                               0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 1.0);
            expect_float4x4_eq!(output[2], 0.0, 0.0, -10.0, 0.0, 0.0, 10.0, 0.0, 0.0,
                               10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 1.0);
            expect_float4x4_eq!(output[3], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 3.0, 4.0, 6.0, 1.0);
            expect_float4x4_eq!(output[4], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 15.0, 50.0, -6.0, 1.0);
            expect_float4x4_eq!(output[5], -0.1, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0,
                               0.0, -0.1, 0.0, 15.0, 50.0, -6.0, 1.0);
            expect_float4x4_eq!(output[6], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 6.0, 8.0, 11.0, 1.0);
            expect_float4x4_eq!(output[7], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        }

        {  // Updates from j7, j0-6 shouldn't be updated
            output.fill(Float4x4::identity());

            let mut job = LocalToModelJob::new();
            job.skeleton = skeleton.as_ref();
            job.from = 7;
            job.input = &input;
            job.output = &mut output;
            assert_eq!(job.validate(), true);
            assert_eq!(job.run(), true);

            expect_float4x4_eq!(output[0], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[1], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[2], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[3], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[4], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[5], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[6], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[7], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 6.0, 7.0, 8.0, 1.0);
        }

        {  // Updates from j1, j1-2 should be updated
            assert_eq!(job_full.run(), true);
            output.fill(Float4x4::identity());

            let mut job = LocalToModelJob::new();
            job.skeleton = skeleton.as_ref();
            job.from = 1;
            job.input = &input;
            job.output = &mut output;
            assert_eq!(job.validate(), true);
            assert_eq!(job.run(), true);

            expect_float4x4_eq!(output[0], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 2.0, 2.0, 2.0, 1.0);
            expect_float4x4_eq!(output[1], 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
                               0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 1.0);
            expect_float4x4_eq!(output[2], 0.0, 0.0, -10.0, 0.0, 0.0, 10.0, 0.0, 0.0,
                               10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 1.0);
            expect_float4x4_eq!(output[3], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[4], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[5], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[6], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[7], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        }

        {  // Updates from j3, j3-6 should be updated
            assert_eq!(job_full.run(), true);
            output.fill(Float4x4::identity());

            let mut job = LocalToModelJob::new();
            job.skeleton = skeleton.as_ref();
            job.from = 3;
            job.input = &input;
            job.output = &mut output;
            assert_eq!(job.validate(), true);
            assert_eq!(job.run(), true);

            expect_float4x4_eq!(output[0], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 2.0, 2.0, 2.0, 1.0);
            expect_float4x4_eq!(output[1], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[2], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[3], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 3.0, 4.0, 6.0, 1.0);
            expect_float4x4_eq!(output[4], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 15.0, 50.0, -6.0, 1.0);
            expect_float4x4_eq!(output[5], -0.1, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0,
                               0.0, -0.1, 0.0, 15.0, 50.0, -6.0, 1.0);
            expect_float4x4_eq!(output[6], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 6.0, 8.0, 11.0, 1.0);
            expect_float4x4_eq!(output[7], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        }

        {  // Updates from j5, j5 should only be updated
            assert_eq!(job_full.run(), true);
            output.fill(Float4x4::identity());

            let mut job = LocalToModelJob::new();
            job.skeleton = skeleton.as_ref();
            job.from = 5;
            job.input = &input;
            job.output = &mut output;
            assert_eq!(job.validate(), true);
            assert_eq!(job.run(), true);

            expect_float4x4_eq!(output[0], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[1], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[2], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[3], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[4], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 15.0, 50.0, -6.0, 1.0);
            expect_float4x4_eq!(output[5], -0.1, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0,
                               0.0, -0.1, 0.0, 15.0, 50.0, -6.0, 1.0);
            expect_float4x4_eq!(output[6], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[7], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        }

        {  // Updates from j6, j6 should only be updated
            assert_eq!(job_full.run(), true);
            output.fill(Float4x4::identity());

            let mut job = LocalToModelJob::new();
            job.skeleton = skeleton.as_ref();
            job.from = 6;
            job.input = &input;
            job.output = &mut output;
            assert_eq!(job.validate(), true);
            assert_eq!(job.run(), true);

            expect_float4x4_eq!(output[0], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[1], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[2], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[3], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 3.0, 4.0, 6.0, 1.0);
            expect_float4x4_eq!(output[4], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[5], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[6], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 6.0, 8.0, 11.0, 1.0);
            expect_float4x4_eq!(output[7], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        }

        {  // Updates from j0 to j2,
            output.fill(Float4x4::identity());

            let mut job = LocalToModelJob::new();
            job.skeleton = skeleton.as_ref();
            job.from = 0;
            job.to = 2;
            job.input = &input;
            job.output = &mut output;
            assert_eq!(job.validate(), true);
            assert_eq!(job.run(), true);

            expect_float4x4_eq!(output[0], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 2.0, 2.0, 2.0, 1.0);
            expect_float4x4_eq!(output[1], 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
                               0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 1.0);
            expect_float4x4_eq!(output[2], 0.0, 0.0, -10.0, 0.0, 0.0, 10.0, 0.0, 0.0,
                               10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 1.0);
            expect_float4x4_eq!(output[3], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[4], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[5], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[6], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[7], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        }

        {  // Updates from j0 to j6, j7 shouldn't be updated
            output.fill(Float4x4::identity());

            let mut job = LocalToModelJob::new();
            job.skeleton = skeleton.as_ref();
            job.from = 0;
            job.to = 6;
            job.input = &input;
            job.output = &mut output;
            assert_eq!(job.validate(), true);
            assert_eq!(job.run(), true);

            expect_float4x4_eq!(output[0], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 2.0, 2.0, 2.0, 1.0);
            expect_float4x4_eq!(output[1], 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
                               0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 1.0);
            expect_float4x4_eq!(output[2], 0.0, 0.0, -10.0, 0.0, 0.0, 10.0, 0.0, 0.0,
                               10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 1.0);
            expect_float4x4_eq!(output[3], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 3.0, 4.0, 6.0, 1.0);
            expect_float4x4_eq!(output[4], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 15.0, 50.0, -6.0, 1.0);
            expect_float4x4_eq!(output[5], -0.1, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0,
                               0.0, -0.1, 0.0, 15.0, 50.0, -6.0, 1.0);
            expect_float4x4_eq!(output[6], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 6.0, 8.0, 11.0, 1.0);
            expect_float4x4_eq!(output[7], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        }

        {  // Updates from j0 to past end, j7 shouldn't be updated
            output.fill(Float4x4::identity());

            let mut job = LocalToModelJob::new();
            job.skeleton = skeleton.as_ref();
            job.from = 0;
            job.to = 46;
            job.input = &input;
            job.output = &mut output;
            assert_eq!(job.validate(), true);
            assert_eq!(job.run(), true);

            expect_float4x4_eq!(output[0], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 2.0, 2.0, 2.0, 1.0);
            expect_float4x4_eq!(output[1], 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
                               0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 1.0);
            expect_float4x4_eq!(output[2], 0.0, 0.0, -10.0, 0.0, 0.0, 10.0, 0.0, 0.0,
                               10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 1.0);
            expect_float4x4_eq!(output[3], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 3.0, 4.0, 6.0, 1.0);
            expect_float4x4_eq!(output[4], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 15.0, 50.0, -6.0, 1.0);
            expect_float4x4_eq!(output[5], -0.1, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0,
                               0.0, -0.1, 0.0, 15.0, 50.0, -6.0, 1.0);
            expect_float4x4_eq!(output[6], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 6.0, 8.0, 11.0, 1.0);
            expect_float4x4_eq!(output[7], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        }

        {  // Updates from j0 to nowehere, nothing should be updated
            output.fill(Float4x4::identity());

            let mut job = LocalToModelJob::new();
            job.skeleton = skeleton.as_ref();
            job.from = 0;
            job.to = -99;
            job.input = &input;
            job.output = &mut output;
            assert_eq!(job.validate(), true);
            assert_eq!(job.run(), true);

            expect_float4x4_eq!(output[0], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[1], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[2], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[3], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[4], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[5], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[6], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[7], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        }

        {  // Updates from out-of-range value, nothing should be updated
            assert_eq!(job_full.run(), true);
            output.fill(Float4x4::identity());

            let mut job = LocalToModelJob::new();
            job.skeleton = skeleton.as_ref();
            job.from = 93;
            job.input = &input;
            job.output = &mut output;
            assert_eq!(job.validate(), true);
            assert_eq!(job.run(), true);

            expect_float4x4_eq!(output[0], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[1], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[2], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[3], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[4], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[5], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[6], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[7], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        }
    }

    #[test]
    fn transformation_from_to_exclude() {
        // Builds the skeleton
        /*
         6 joints
               *
             /   \
           j0    j7
          /  \
         j1  j3
          |  / \
         j2 j4 j6
             |
            j5
        */
        let mut raw_skeleton = RawSkeleton::new();
        raw_skeleton.roots.resize(2, Joint::new());
        let j0 = &mut raw_skeleton.roots[0];
        j0.name = "j0".to_string();
        let j7 = &mut raw_skeleton.roots[1];
        j7.name = "j7".to_string();

        j0.children.resize(2, Joint::new());
        j0.children[0].name = "j1".to_string();
        j0.children[1].name = "j3".to_string();

        j0.children[0].children.resize(1, Joint::new());
        j0.children[0].children[0].name = "j2".to_string();

        j0.children[1].children.resize(2, Joint::new());
        j0.children[1].children[0].name = "j4".to_string();
        j0.children[1].children[1].name = "j6".to_string();

        j0.children[1].children[0].children.resize(1, Joint::new());
        j0.children[1].children[0].children[0].name = "j5".to_string();

        assert_eq!(raw_skeleton.validate(), true);
        assert_eq!(raw_skeleton.num_joints(), 8);

        let skeleton = SkeletonBuilder::apply(&raw_skeleton);
        assert_eq!(skeleton.is_some(), true);

        // Initializes an input transformation.
        let input = [
            // Stores up to 8 inputs, needs 7.
            //                             j0   j1   j2    j3
            SoaTransform {
                translation: SoaFloat3::load(SimdFloat4::load(2.0, 0.0, -2.0, 1.0),
                                             SimdFloat4::load(2.0, 0.0, -2.0, 2.0),
                                             SimdFloat4::load(2.0, 0.0, -2.0, 4.0)),
                rotation: SoaQuaternion::load(SimdFloat4::load(0.0, 0.0, 0.0, 0.0),
                                              SimdFloat4::load(0.0, 0.70710677, 0.0, 0.0),
                                              SimdFloat4::load(0.0, 0.0, 0.0, 0.0),
                                              SimdFloat4::load(1.0, 0.70710677, 1.0, 1.0)),
                scale: SoaFloat3::load(SimdFloat4::load(1.0, 1.0, 10.0, 1.0),
                                       SimdFloat4::load(1.0, 1.0, 10.0, 1.0),
                                       SimdFloat4::load(1.0, 1.0, 10.0, 1.0)),
            },
            //                             j4    j5   j6   j7.
            SoaTransform {
                translation: SoaFloat3::load(SimdFloat4::load(12.0, 0.0, 3.0, 6.0),
                                             SimdFloat4::load(46.0, 0.0, 4.0, 7.0),
                                             SimdFloat4::load(-12.0, 0.0, 5.0, 8.0)),
                rotation: SoaQuaternion::load(SimdFloat4::load(0.0, 0.0, 0.0, 0.0),
                                              SimdFloat4::load(0.0, 0.0, 0.0, 0.0),
                                              SimdFloat4::load(0.0, 0.0, 0.0, 0.0),
                                              SimdFloat4::load(1.0, 1.0, 1.0, 1.0)),
                scale: SoaFloat3::load(SimdFloat4::load(1.0, -0.1, 1.0, 1.0),
                                       SimdFloat4::load(1.0, -0.1, 1.0, 1.0),
                                       SimdFloat4::load(1.0, -0.1, 1.0, 1.0)),
            }];

        let mut output = [Float4x4::identity(); 8];
        let mut job_full = LocalToModelJob::new();
        {  // Intialize whole hierarchy output
            job_full.skeleton = skeleton.as_ref();
            job_full.from = crate::skeleton::Constants::KNoParent as i32;
            job_full.from_excluded = true;
            job_full.input = &input;
            job_full.output = &mut output;
            assert_eq!(job_full.validate(), true);
            assert_eq!(job_full.run(), true);
            expect_float4x4_eq!(output[0], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 2.0, 2.0, 2.0, 1.0);
            expect_float4x4_eq!(output[1], 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
                               0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 1.0);
            expect_float4x4_eq!(output[2], 0.0, 0.0, -10.0, 0.0, 0.0, 10.0, 0.0, 0.0,
                               10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 1.0);
            expect_float4x4_eq!(output[3], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 3.0, 4.0, 6.0, 1.0);
            expect_float4x4_eq!(output[4], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 15.0, 50.0, -6.0, 1.0);
            expect_float4x4_eq!(output[5], -0.1, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0,
                               0.0, -0.1, 0.0, 15.0, 50.0, -6.0, 1.0);
            expect_float4x4_eq!(output[6], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 6.0, 8.0, 11.0, 1.0);
            expect_float4x4_eq!(output[7], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 6.0, 7.0, 8.0, 1.0);
        }

        {  // Updates from j0 excluded, j7 shouldn't be updated
            output.fill(Float4x4::identity());
            output[0] = Float4x4::scaling(SimdFloat4::load(2.0, 2.0, 2.0, 0.0));

            let mut job = LocalToModelJob::new();
            job.skeleton = skeleton.as_ref();
            job.from = 0;
            job.from_excluded = true;
            job.input = &input;
            job.output = &mut output;
            assert_eq!(job.validate(), true);
            assert_eq!(job.run(), true);

            expect_float4x4_eq!(output[0], 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0,
                               0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[1], 0.0, 0.0, -2.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0,
                               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[2], 0.0, 0.0, -20.0, 0.0, 0.0, 20.0, 0.0, 0.0,
                               20.0, 0.0, 0.0, 0.0, -4.0, -4.0, 4.0, 1.0);
            expect_float4x4_eq!(output[3], 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0,
                               0.0, 2.0, 0.0, 2.0, 4.0, 8.0, 1.0);
            expect_float4x4_eq!(output[4], 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0,
                               0.0, 2.0, 0.0, 26.0, 96.0, -16.0, 1.0);
            expect_float4x4_eq!(output[5], -0.2, 0.0, 0.0, 0.0, 0.0, -0.2, 0.0, 0.0, 0.0,
                               0.0, -0.2, 0.0, 26.0, 96.0, -16.0, 1.0);
            expect_float4x4_eq!(output[6], 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0,
                               0.0, 2.0, 0.0, 8.0, 12.0, 18.0, 1.0);
            expect_float4x4_eq!(output[7], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        }

        {  // Updates from j1 exclude, j2 should be updated
            assert_eq!(job_full.run(), true);
            output.fill(Float4x4::identity());
            output[1] = Float4x4::scaling(SimdFloat4::load(2.0, 2.0, 2.0, 0.0));

            let mut job = LocalToModelJob::new();
            job.skeleton = skeleton.as_ref();
            job.from = 1;
            job.from_excluded = true;
            job.input = &input;
            job.output = &mut output;
            assert_eq!(job.validate(), true);
            assert_eq!(job.run(), true);

            expect_float4x4_eq!(output[0], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[1], 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0,
                               0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[2], 20.0, 0.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0,
                               0.0, 20.0, 0.0, -4.0, -4.0, -4.0, 1.0);
            expect_float4x4_eq!(output[3], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[4], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[5], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[6], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[7], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        }

        {  // Updates from j2 excluded, no joint should be updated
            assert_eq!(job_full.run(), true);
            output.fill(Float4x4::identity());
            output[2] = Float4x4::scaling(SimdFloat4::load(2.0, 2.0, 2.0, 0.0));

            let mut job = LocalToModelJob::new();
            job.skeleton = skeleton.as_ref();
            job.from = 2;
            job.from_excluded = true;
            job.input = &input;
            job.output = &mut output;
            assert_eq!(job.validate(), true);
            assert_eq!(job.run(), true);

            expect_float4x4_eq!(output[0], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[1], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[2], 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0,
                               0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[3], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[4], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[5], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[6], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[7], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        }

        {  // Updates from j7 excluded, no joint should be updated
            output.fill(Float4x4::identity());
            output[7] = Float4x4::scaling(SimdFloat4::load(2.0, 2.0, 2.0, 0.0));

            let mut job = LocalToModelJob::new();
            job.skeleton = skeleton.as_ref();
            job.from = 7;
            job.from_excluded = true;
            job.input = &input;
            job.output = &mut output;
            assert_eq!(job.validate(), true);
            assert_eq!(job.run(), true);

            expect_float4x4_eq!(output[0], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[1], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[2], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[3], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[4], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[5], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[6], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[7], 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0,
                               0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        }

        {  // Updates from j6 excluded, no joint should be updated
            assert_eq!(job_full.run(), true);
            output.fill(Float4x4::identity());
            output[6] = Float4x4::scaling(SimdFloat4::load(2.0, 2.0, 2.0, 0.0));

            let mut job = LocalToModelJob::new();
            job.skeleton = skeleton.as_ref();
            job.from = 6;
            job.from_excluded = true;
            job.input = &input;
            job.output = &mut output;
            assert_eq!(job.validate(), true);
            assert_eq!(job.run(), true);

            expect_float4x4_eq!(output[0], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[1], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[2], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[3], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[4], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[5], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[6], 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0,
                               0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0);
            expect_float4x4_eq!(output[7], 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                               0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        }
    }

    #[test]
    fn empty() {
        let skeleton = Skeleton::new();

        {  // From root
            let mut job = LocalToModelJob::new();
            job.skeleton = Some(&skeleton);
            assert_eq!(job.validate(), true);
            assert_eq!(job.run(), true);
        }

        {  // From joint 0
            let mut job = LocalToModelJob::new();
            job.from = 0;
            job.skeleton = Some(&skeleton);
            assert_eq!(job.validate(), true);
            assert_eq!(job.run(), true);
        }
    }
}