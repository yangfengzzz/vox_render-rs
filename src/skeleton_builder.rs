/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use crate::raw_skeleton::*;
use crate::skeleton::Skeleton;
use crate::simd_math::SimdFloat4;

// Defines the class responsible of building Skeleton instances.
pub struct SkeletonBuilder {}

impl SkeletonBuilder {
    // Creates a Skeleton based on _raw_skeleton and *this builder parameters.
    // Returns a Skeleton instance on success, an empty unique_ptr on failure. See
    // RawSkeleton::Validate() for more details about failure reasons.
    // The skeleton is returned as an unique_ptr as ownership is given back to the
    // caller.
    pub fn apply(_raw_skeleton: &RawSkeleton) -> Option<Skeleton> {
        // Tests _raw_skeleton validity.
        if !_raw_skeleton.validate() {
            return None;
        }

        // step1: Everything is fine, allocates and fills the skeleton.
        // Will not fail.
        let mut skeleton = Skeleton::new();
        let num_joints = _raw_skeleton.num_joints();

        // step2: Iterates through all the joint of the raw skeleton and fills a sorted joint list.
        // Iteration order defines runtime skeleton joint ordering.
        let mut lister = JointLister::new(num_joints);
        let linear_joints = &iterate_joints_df(_raw_skeleton, &mut lister).linear_joints;
        debug_assert!(linear_joints.len() as i32 == num_joints);

        // step3: Allocates all skeleton members.
        skeleton.allocate(num_joints as usize);

        // step4: Copy names
        for i in 0..num_joints as usize {
            let current = linear_joints[i].joint;
            skeleton.joint_names_[i] = current.name.clone();
        }

        // step5: Transfers sorted joints hierarchy to the new skeleton.
        for i in 0..num_joints as usize {
            skeleton.joint_parents_[i] = linear_joints[i].parent;
        }

        // step6: Transfers t-poses.
        let w_axis = SimdFloat4::w_axis();
        let zero = SimdFloat4::zero();
        let one = SimdFloat4::one();

        for i in 0..skeleton.num_soa_joints() as usize {
            let mut translations = [SimdFloat4::zero(); 4];
            let mut scales = [SimdFloat4::zero(); 4];
            let mut rotations = [SimdFloat4::zero(); 4];
            for j in 0..4_usize {
                if i * 4 + j < num_joints as usize {
                    let src_joint = linear_joints[i * 4 + j].joint;
                    translations[j] = SimdFloat4::load3ptr_u(src_joint.transform.translation.to_vec4());
                    rotations[j] = SimdFloat4::load_ptr_u(src_joint.transform.rotation.to_vec()).normalize_safe4(w_axis);
                    scales[j] = SimdFloat4::load3ptr_u(src_joint.transform.scale.to_vec4());
                } else {
                    translations[j] = zero;
                    rotations[j] = w_axis;
                    scales[j] = one;
                }
            }
            // Fills the SoaTransform structure.
            SimdFloat4::transpose4x3(&translations,
                                     &mut skeleton.joint_bind_poses_[i].translation);
            SimdFloat4::transpose4x4(&rotations, &mut skeleton.joint_bind_poses_[i].rotation);
            SimdFloat4::transpose4x3(&scales, &mut skeleton.joint_bind_poses_[i].scale);
        }

        return Some(skeleton);  // Success.
    }
}

//--------------------------------------------------------------------------------------------------
struct Joint<'a> {
    joint: &'a crate::raw_skeleton::Joint,
    parent: i16,
}

// Stores each traversed joint in a vector.
struct JointLister<'a> {
    // Array of joints in the traversed DAG order.
    linear_joints: Vec<Joint<'a>>,
}

impl<'a> JointLister<'a> {
    fn new(_num_joints: i32) -> JointLister<'a> {
        let mut result = JointLister {
            linear_joints: vec![]
        };
        result.linear_joints.reserve(_num_joints as usize);
        return result;
    }
}

impl<'a> SkeletonVisitor<'a> for JointLister<'a> {
    fn visitor(&mut self, _current: &'a crate::raw_skeleton::Joint,
               _parent: Option<&crate::raw_skeleton::Joint>) {
        // Looks for the "lister" parent.
        let mut parent = crate::skeleton::Constants::KNoParent as i16;
        if _parent.is_some() {
            // Start searching from the last joint.
            let mut j = self.linear_joints.len() as i16 - 1;
            while j >= 0 {
                if self.linear_joints[j as usize].joint as *const _ == _parent.unwrap() as *const _ {
                    parent = j;
                    break;
                }
                j -= 1;
            }
            debug_assert!(parent >= 0);
        }
        let listed = Joint { joint: _current, parent };
        self.linear_joints.push(listed);
    }
}

//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------
#[cfg(test)]
mod skeleton_builder {
    use crate::skeleton_builder::SkeletonBuilder;
    use crate::raw_skeleton::*;

    #[test]
    fn error() {
        // The default raw skeleton is valid. It has no joint.
        {
            let raw_skeleton = RawSkeleton::new();
            assert_eq!(raw_skeleton.validate(), true);
            assert_eq!(raw_skeleton.num_joints(), 0);

            let skeleton = SkeletonBuilder::apply(&raw_skeleton);
            assert_eq!(skeleton.is_some(), true);
            assert_eq!(skeleton.unwrap().num_joints(), 0);
        }
    }

    // Tester object that ensure order (depth-first) of joints iteration.
    struct RawSkeletonIterateDFTester {
        num_joint_: i32,
    }

    impl RawSkeletonIterateDFTester {
        fn new() -> RawSkeletonIterateDFTester {
            return RawSkeletonIterateDFTester {
                num_joint_: 0
            };
        }
    }

    impl<'a> SkeletonVisitor<'a> for RawSkeletonIterateDFTester {
        fn visitor(&mut self, _current: &'a Joint, _parent: Option<&Joint>) {
            match self.num_joint_ {
                0 => {
                    assert_eq!(_current.name == "root" && _parent.is_none(), true);
                }
                1 => {
                    assert_eq!(_current.name == "j0" && _parent.unwrap().name == "root", true);
                }
                2 => {
                    assert_eq!(_current.name == "j1" && _parent.unwrap().name == "root", true);
                }
                3 => {
                    assert_eq!(_current.name == "j2" && _parent.unwrap().name == "j1", true);
                }
                4 => {
                    assert_eq!(_current.name == "j3" && _parent.unwrap().name == "j1", true);
                }
                5 => {
                    assert_eq!(_current.name == "j4" && _parent.unwrap().name == "root", true);
                }
                _ => {
                    assert_eq!(true, false)
                }
            }
            self.num_joint_ += 1;
        }
    }

    // Tester object that ensure order (breadth-first) of joints iteration.
    struct RawSkeletonIterateBFTester {
        num_joint_: i32,
    }

    impl RawSkeletonIterateBFTester {
        fn new() -> RawSkeletonIterateBFTester {
            return RawSkeletonIterateBFTester {
                num_joint_: 0
            };
        }
    }

    impl<'a> SkeletonVisitor<'a> for RawSkeletonIterateBFTester {
        fn visitor(&mut self, _current: &'a Joint, _parent: Option<&Joint>) {
            match self.num_joint_ {
                0 => {
                    assert_eq!(_current.name == "root" && _parent.is_none(), true);
                }
                1 => {
                    assert_eq!(_current.name == "j0" && _parent.unwrap().name == "root", true);
                }
                2 => {
                    assert_eq!(_current.name == "j1" && _parent.unwrap().name == "root", true);
                }
                3 => {
                    assert_eq!(_current.name == "j4" && _parent.unwrap().name == "root", true);
                }
                4 => {
                    assert_eq!(_current.name == "j2" && _parent.unwrap().name == "j1", true);
                }
                5 => {
                    assert_eq!(_current.name == "j3" && _parent.unwrap().name == "j1", true);
                }
                _ => {
                    assert_eq!(true, false)
                }
            }
            self.num_joint_ += 1;
        }
    }

    #[test]
    fn skeleton_builder() {
        /*
        5 joints

           *
           |
          root
          / |  \
         j0 j1 j4
            / \
           j2 j3
        */
        let mut raw_skeleton = RawSkeleton::new();
        raw_skeleton.roots.resize(1, Joint::new());
        let root = &mut raw_skeleton.roots[0];
        root.name = "root".to_string();

        root.children.resize(3, Joint::new());
        root.children[0].name = "j0".to_string();
        root.children[1].name = "j1".to_string();
        root.children[2].name = "j4".to_string();

        root.children[1].children.resize(2, Joint::new());
        root.children[1].children[0].name = "j2".to_string();
        root.children[1].children[1].name = "j3".to_string();

        assert_eq!(raw_skeleton.validate(), true);
        assert_eq!(raw_skeleton.num_joints(), 6);

        iterate_joints_df(&raw_skeleton, &mut RawSkeletonIterateDFTester::new());
        iterate_joints_bf(&raw_skeleton, &mut RawSkeletonIterateBFTester::new());
    }

    #[test]
    fn build() {
        // 1 joint: the root.
        {
            let mut raw_skeleton = RawSkeleton::new();
            raw_skeleton.roots.resize(1, Joint::new());
            let root = &mut raw_skeleton.roots[0];
            root.name = "root".to_string();

            assert_eq!(raw_skeleton.validate(), true);
            assert_eq!(raw_skeleton.num_joints(), 1);

            let skeleton = SkeletonBuilder::apply(&raw_skeleton);
            assert_eq!(skeleton.is_some(), true);
            assert_eq!(skeleton.as_ref().unwrap().num_joints(), 1);
            assert_eq!(skeleton.as_ref().unwrap().joint_parents()[0], crate::skeleton::Constants::KNoParent as i16);
        }

        /*
         2 joints

           *
           |
          j0
           |
          j1
        */
        {
            let mut raw_skeleton = RawSkeleton::new();
            raw_skeleton.roots.resize(1, Joint::new());
            let root = &mut raw_skeleton.roots[0];
            root.name = "j0".to_string();

            root.children.resize(1, Joint::new());
            root.children[0].name = "j1".to_string();

            assert_eq!(raw_skeleton.validate(), true);
            assert_eq!(raw_skeleton.num_joints(), 2);

            let skeleton = SkeletonBuilder::apply(&raw_skeleton);
            assert_eq!(skeleton.is_some(), true);
            assert_eq!(skeleton.as_ref().unwrap().num_joints(), 2);
            for i in 0..skeleton.as_ref().unwrap().num_joints() as usize {
                let parent_index = skeleton.as_ref().unwrap().joint_parents()[i] as usize;
                if skeleton.as_ref().unwrap().joint_names()[i] == "j0" {
                    assert_eq!(parent_index, crate::skeleton::Constants::KNoParent as usize);
                } else if skeleton.as_ref().unwrap().joint_names()[i] == "j1" {
                    assert_eq!(skeleton.as_ref().unwrap().joint_names()[parent_index], "j0");
                } else {
                    assert_eq!(false, true);
                }
            }
        }

        /*
         4 joints

           *
           |
          j0
          / \
         j1 j3
          |
         j2
        */
        {
            let mut raw_skeleton = RawSkeleton::new();
            raw_skeleton.roots.resize(1, Joint::new());
            let root = &mut raw_skeleton.roots[0];
            root.name = "j0".to_string();

            root.children.resize(2, Joint::new());
            root.children[0].name = "j1".to_string();
            root.children[1].name = "j3".to_string();

            root.children[0].children.resize(1, Joint::new());
            root.children[0].children[0].name = "j2".to_string();

            assert_eq!(raw_skeleton.validate(), true);
            assert_eq!(raw_skeleton.num_joints(), 4);

            let skeleton = SkeletonBuilder::apply(&raw_skeleton);
            assert_eq!(skeleton.is_some(), true);
            assert_eq!(skeleton.as_ref().unwrap().num_joints(), 4);
            for i in 0..skeleton.as_ref().unwrap().num_joints() as usize {
                let parent_index = skeleton.as_ref().unwrap().joint_parents()[i] as usize;
                if skeleton.as_ref().unwrap().joint_names()[i] == "j0" {
                    assert_eq!(parent_index, crate::skeleton::Constants::KNoParent as usize);
                } else if skeleton.as_ref().unwrap().joint_names()[i] == "j1" {
                    assert_eq!(skeleton.as_ref().unwrap().joint_names()[parent_index], "j0");
                } else if skeleton.as_ref().unwrap().joint_names()[i] == "j2" {
                    assert_eq!(skeleton.as_ref().unwrap().joint_names()[parent_index], "j1");
                } else if skeleton.as_ref().unwrap().joint_names()[i] == "j3" {
                    assert_eq!(skeleton.as_ref().unwrap().joint_names()[parent_index], "j0");
                } else {
                    assert_eq!(false, true);
                }
            }
        }


        /*
         4 joints

           *
           |
          j0
          / \
         j1 j2
             |
            j3
        */
        {
            let mut raw_skeleton = RawSkeleton::new();
            raw_skeleton.roots.resize(1, Joint::new());
            let root = &mut raw_skeleton.roots[0];
            root.name = "j0".to_string();

            root.children.resize(2, Joint::new());
            root.children[0].name = "j1".to_string();
            root.children[1].name = "j2".to_string();

            root.children[1].children.resize(1, Joint::new());
            root.children[1].children[0].name = "j3".to_string();

            assert_eq!(raw_skeleton.validate(), true);
            assert_eq!(raw_skeleton.num_joints(), 4);

            let skeleton = SkeletonBuilder::apply(&raw_skeleton);
            assert_eq!(skeleton.is_some(), true);
            assert_eq!(skeleton.as_ref().unwrap().num_joints(), 4);
            for i in 0..skeleton.as_ref().unwrap().num_joints() as usize {
                let parent_index = skeleton.as_ref().unwrap().joint_parents()[i] as usize;
                if skeleton.as_ref().unwrap().joint_names()[i] == "j0" {
                    assert_eq!(parent_index, crate::skeleton::Constants::KNoParent as usize);
                } else if skeleton.as_ref().unwrap().joint_names()[i] == "j1" {
                    assert_eq!(skeleton.as_ref().unwrap().joint_names()[parent_index], "j0");
                } else if skeleton.as_ref().unwrap().joint_names()[i] == "j2" {
                    assert_eq!(skeleton.as_ref().unwrap().joint_names()[parent_index], "j0");
                } else if skeleton.as_ref().unwrap().joint_names()[i] == "j3" {
                    assert_eq!(skeleton.as_ref().unwrap().joint_names()[parent_index], "j2");
                } else {
                    assert_eq!(false, true);
                }
            }
        }


        /*
         5 joints

           *
           |
          j0
          / \
         j1 j2
            / \
           j3 j4
        */
        {
            let mut raw_skeleton = RawSkeleton::new();
            raw_skeleton.roots.resize(1, Joint::new());
            let root = &mut raw_skeleton.roots[0];
            root.name = "j0".to_string();

            root.children.resize(2, Joint::new());
            root.children[0].name = "j1".to_string();
            root.children[1].name = "j2".to_string();

            root.children[1].children.resize(2, Joint::new());
            root.children[1].children[0].name = "j3".to_string();
            root.children[1].children[1].name = "j4".to_string();

            assert_eq!(raw_skeleton.validate(), true);
            assert_eq!(raw_skeleton.num_joints(), 5);

            let skeleton = SkeletonBuilder::apply(&raw_skeleton);
            assert_eq!(skeleton.is_some(), true);
            assert_eq!(skeleton.as_ref().unwrap().num_joints(), 5);
            for i in 0..skeleton.as_ref().unwrap().num_joints() as usize {
                let parent_index = skeleton.as_ref().unwrap().joint_parents()[i] as usize;
                if skeleton.as_ref().unwrap().joint_names()[i] == "j0" {
                    assert_eq!(parent_index, crate::skeleton::Constants::KNoParent as usize);
                } else if skeleton.as_ref().unwrap().joint_names()[i] == "j1" {
                    assert_eq!(skeleton.as_ref().unwrap().joint_names()[parent_index], "j0");
                } else if skeleton.as_ref().unwrap().joint_names()[i] == "j2" {
                    assert_eq!(skeleton.as_ref().unwrap().joint_names()[parent_index], "j0");
                } else if skeleton.as_ref().unwrap().joint_names()[i] == "j3" {
                    assert_eq!(skeleton.as_ref().unwrap().joint_names()[parent_index], "j2");
                } else if skeleton.as_ref().unwrap().joint_names()[i] == "j4" {
                    assert_eq!(skeleton.as_ref().unwrap().joint_names()[parent_index], "j2");
                } else {
                    assert_eq!(false, true);
                }
            }
        }

        /*
         6 joints

           *
           |
          j0
          /  \
         j1  j3
          |  / \
         j2 j4 j5
        */
        {
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
            assert_eq!(skeleton.as_ref().unwrap().num_joints(), 6);
            for i in 0..skeleton.as_ref().unwrap().num_joints() as usize {
                let parent_index = skeleton.as_ref().unwrap().joint_parents()[i] as usize;
                if skeleton.as_ref().unwrap().joint_names()[i] == "j0" {
                    assert_eq!(parent_index, crate::skeleton::Constants::KNoParent as usize);
                } else if skeleton.as_ref().unwrap().joint_names()[i] == "j1" {
                    assert_eq!(skeleton.as_ref().unwrap().joint_names()[parent_index], "j0");
                } else if skeleton.as_ref().unwrap().joint_names()[i] == "j2" {
                    assert_eq!(skeleton.as_ref().unwrap().joint_names()[parent_index], "j1");
                } else if skeleton.as_ref().unwrap().joint_names()[i] == "j3" {
                    assert_eq!(skeleton.as_ref().unwrap().joint_names()[parent_index], "j0");
                } else if skeleton.as_ref().unwrap().joint_names()[i] == "j4" {
                    assert_eq!(skeleton.as_ref().unwrap().joint_names()[parent_index], "j3");
                } else if skeleton.as_ref().unwrap().joint_names()[i] == "j5" {
                    assert_eq!(skeleton.as_ref().unwrap().joint_names()[parent_index], "j3");
                } else {
                    assert_eq!(false, true);
                }
            }

            // Skeleton joins should be sorted "per parent" and maintain original
            // children joint order.
            assert_eq!(skeleton.as_ref().unwrap().joint_parents()[0], crate::skeleton::Constants::KNoParent as i16);
            assert_eq!(skeleton.as_ref().unwrap().joint_names()[0], "j0");
            assert_eq!(skeleton.as_ref().unwrap().joint_parents()[1], 0);
            assert_eq!(skeleton.as_ref().unwrap().joint_names()[1], "j1");
            assert_eq!(skeleton.as_ref().unwrap().joint_parents()[2], 1);
            assert_eq!(skeleton.as_ref().unwrap().joint_names()[2], "j2");
            assert_eq!(skeleton.as_ref().unwrap().joint_parents()[3], 0);
            assert_eq!(skeleton.as_ref().unwrap().joint_names()[3], "j3");
            assert_eq!(skeleton.as_ref().unwrap().joint_parents()[4], 3);
            assert_eq!(skeleton.as_ref().unwrap().joint_names()[4], "j4");
            assert_eq!(skeleton.as_ref().unwrap().joint_parents()[5], 3);
            assert_eq!(skeleton.as_ref().unwrap().joint_names()[5], "j5");
        }
    }
}