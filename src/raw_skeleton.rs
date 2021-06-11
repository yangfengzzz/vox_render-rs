/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use crate::transform::Transform;

// Offline skeleton joint type.
pub struct Joint {
    // Children joints.
    pub children: Vec<Joint>,

    // The name of the joint.
    pub name: String,

    // Joint bind pose transformation in local space.
    pub transform: Transform,
}

// Off-line skeleton type.
// This skeleton type is not intended to be used in run time. It is used to
// define the offline skeleton object that can be converted to the runtime
// skeleton using the SkeletonBuilder. This skeleton structure exposes joints'
// hierarchy. A joint is defined with a name, a transformation (its bind pose),
// and its children. Children are exposed as a public std::vector of joints.
// This same type is used for skeleton roots, also exposed from the public API.
// The public API exposed through std:vector's of joints can be used freely with
// the only restriction that the total number of joints does not exceed
// Skeleton::kMaxJoints.
pub struct RawSkeleton {
    // Declares the skeleton's roots. Can be empty if the skeleton has no joint.
    pub roots: Vec<Joint>,
}

impl RawSkeleton {
    // Construct an empty skeleton.
    pub fn new() -> RawSkeleton {
        return RawSkeleton {
            roots: vec![]
        };
    }

    // Tests for *this validity.
    // Returns true on success or false on failure if the number of joints exceeds
    // ozz::Skeleton::kMaxJoints.
    pub fn validate(&self) -> bool {
        if self.num_joints() > crate::skeleton::Constants::KMaxJoints as i32 {
            return false;
        }
        return true;
    }

    // Returns the number of joints of *this animation.
    // This function is not constant time as it iterates the hierarchy of joints
    // and counts them.
    pub fn num_joints(&self) -> i32 {
        struct JointCounter {
            num_joints: i32,
        }
        impl JointCounter {
            fn new() -> JointCounter {
                return JointCounter {
                    num_joints: 0
                };
            }
        }
        impl SkeletonVisitor for JointCounter {
            fn visitor(&mut self, _current: &Joint, _parent: Option<&Joint>) {
                self.num_joints += 1;
            }
        }

        return iterate_joints_df(self, &mut JointCounter::new()).num_joints;
    }
}

//--------------------------------------------------------------------------------------------------
pub trait SkeletonVisitor {
    fn visitor(&mut self, _current: &Joint, _parent: Option<&Joint>);
}

// Internal function used to iterate through joint hierarchy depth-first.
pub fn _iter_hierarchy_recurse_df<_Fct: SkeletonVisitor>(_children: &Vec<Joint>,
                                                         _parent: Option<&Joint>,
                                                         _fct: &mut _Fct) {
    for i in 0.._children.len() {
        let current = &_children[i];
        _fct.visitor(current, _parent);
        _iter_hierarchy_recurse_df(&current.children, Some(current), _fct);
    }
}

// Internal function used to iterate through joint hierarchy breadth-first.
pub fn _iter_hierarchy_recurse_bf<_Fct: SkeletonVisitor>(_children: &Vec<Joint>,
                                                         _parent: Option<&Joint>,
                                                         _fct: &mut _Fct) {
    for i in 0.._children.len() {
        let current = &_children[i];
        _fct.visitor(current, _parent);
    }
    for i in 0.._children.len() {
        let current = &_children[i];
        _iter_hierarchy_recurse_bf(&current.children, Some(current), _fct);
    }
}

// Applies a specified functor to each joint in a depth-first order.
// _Fct is of type void(const Joint& _current, const Joint* _parent) where the
// first argument is the child of the second argument. _parent is null if the
// _current joint is the root.
pub fn iterate_joints_df<'a, _Fct: SkeletonVisitor>(_skeleton: &RawSkeleton, _fct: &'a mut _Fct) -> &'a _Fct {
    _iter_hierarchy_recurse_df(&_skeleton.roots, None, _fct);
    return _fct;
}

// Applies a specified functor to each joint in a breadth-first order.
// _Fct is of type void(const Joint& _current, const Joint* _parent) where the
// first argument is the child of the second argument. _parent is null if the
// _current joint is the root.
pub fn iterate_joints_bf<'a, _Fct: SkeletonVisitor>(_skeleton: &RawSkeleton, _fct: &'a mut _Fct) -> &'a _Fct {
    _iter_hierarchy_recurse_bf(&_skeleton.roots, None, _fct);
    return _fct;
}