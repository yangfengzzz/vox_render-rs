/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use std::collections::BTreeMap;
use crate::raw_animation::{RawAnimation, TranslationKey, RotationKey, ScaleKey};
use crate::skeleton::Skeleton;
use crate::skeleton_utils::{JointVisitor, iterate_joints_df, iterate_joints_df_reverse};
use crate::raw_animation_utils::{lerp_translation, lerp_rotation, lerp_scale};

// Optimization settings.
#[derive(Clone)]
pub struct Setting {
    // The maximum error that an optimization is allowed to generate on a whole
    // joint hierarchy.
    pub tolerance: f32,

    // The distance (from the joint) at which error is measured (if bigger that
    // joint hierarchy). This allows to emulate effect on skinning.
    pub distance: f32,
}

impl Setting {
    // Default settings
    pub fn new_default() -> Setting {
        return Setting {
            tolerance: 1e-3,// 1mm
            distance: 1e-1,// 10cm
        };
    }

    pub fn new(_tolerance: f32, _distance: f32) -> Setting {
        return Setting {
            tolerance: _tolerance,
            distance: _distance,
        };
    }
}

// Defines the class responsible of optimizing an offline raw animation
// instance. Optimization is performed using a key frame reduction technique. It
// decimates redundant / interpolatable key frames, within error tolerances given
// as input. The optimizer takes into account for each joint the error
// generated on its whole child hierarchy. This allows for example to take into
// consideration the error generated on a finger when optimizing the shoulder. A
// small error on the shoulder can be magnified when propagated to the finger
// indeed.
// It's possible to override optimization settings for a joint. This implicitly
// have an effect on the whole chain, up to that joint. This allows for example
// to have aggressive optimization for a whole skeleton, except for the chain
// that leads to the hand if user wants it to be precise. Default optimization
// tolerances are set in order to favor quality over runtime performances and
// memory footprint.
pub struct AnimationOptimizer {
    // Global optimization settings. These settings apply to all joints of the
    // hierarchy, unless overridden by joint specific settings.
    pub setting: Setting,

    // Per joint override of optimization settings.
    pub joints_setting_override: BTreeMap<i32, Setting>,
}

impl AnimationOptimizer {
    // Initializes the optimizer with default tolerances (favoring quality).
    pub fn new() -> AnimationOptimizer {
        return AnimationOptimizer {
            setting: Setting::new_default(),
            joints_setting_override: Default::default(),
        };
    }

    // Optimizes _input using *this parameters. _skeleton is required to evaluate
    // optimization error along joint hierarchy (see hierarchical_tolerance).
    // Returns true on success and fills _output animation with the optimized
    // version of _input animation.
    // *_output must be a valid RawAnimation instance.
    // Returns false on failure and resets _output to an empty animation.
    // See RawAnimation::Validate() for more details about failure reasons.
    pub fn apply(_input: &RawAnimation, _skeleton: &Skeleton, _output: &mut RawAnimation) -> bool {
        todo!()
    }
}

//--------------------------------------------------------------------------------------------------
fn get_joint_setting(_optimizer: &AnimationOptimizer, _joint: i32) -> Setting {
    let mut setting = _optimizer.setting.clone();
    let it = _optimizer.joints_setting_override.iter().find(|x| {
        *(x.0) == _joint
    });
    if it.is_some() {
        setting = it.unwrap().1.clone();
    }

    return setting;
}

#[derive(Clone)]
struct Spec {
    // Length of a joint hierarchy (max of all child).
    length: f32,
    // Scale of a joint hierarchy (accumulated from all parents).
    scale: f32,
    // Tolerance of a joint hierarchy (min of all child).
    tolerance: f32,
}

impl Spec {
    fn new() -> Spec {
        return Spec {
            length: 0.0,
            scale: 0.0,
            tolerance: 0.0,
        };
    }
}

//--------------------------------------------------------------------------------------------------
struct HierarchyBuilder<'a> {
    // Defines the length of a joint hierarchy (of all child).
    specs: Vec<Spec>,

    // Targeted animation.
    animation: &'a RawAnimation,

    // Useful to access settings and compute hierarchy length.
    optimizer: &'a AnimationOptimizer,
}

impl<'a> HierarchyBuilder<'a> {
    fn new(_animation: &'a RawAnimation, _skeleton: &'a Skeleton,
           _optimizer: &'a AnimationOptimizer) -> HierarchyBuilder<'a> {
        let mut builder = HierarchyBuilder {
            specs: vec![],
            animation: _animation,
            optimizer: _optimizer,
        };

        builder.specs.resize(_animation.tracks.len(), Spec::new());
        debug_assert!(_animation.num_tracks() == _skeleton.num_joints());

        // Computes hierarchical scale, iterating skeleton forward (root to leaf).
        iterate_joints_df(_skeleton, ComputeScaleForward::new(&mut builder.specs, &builder.animation, &builder.optimizer), None);

        // Computes hierarchical length, iterating skeleton backward (leaf to root).
        iterate_joints_df_reverse(_skeleton, ComputeLengthBackward::new(&mut builder.specs, &builder.animation, &builder.optimizer));

        return builder;
    }
}

//--------------------------------------------------------------------------------------------------
// Extracts maximum translations and scales for each track/joint.
struct ComputeScaleForward<'a> {
    // Defines the length of a joint hierarchy (of all child).
    specs: &'a mut Vec<Spec>,

    // Targeted animation.
    animation: &'a RawAnimation,

    // Useful to access settings and compute hierarchy length.
    optimizer: &'a AnimationOptimizer,
}

impl<'a> ComputeScaleForward<'a> {
    fn new(specs: &'a mut Vec<Spec>, animation: &'a RawAnimation,
           optimizer: &'a AnimationOptimizer) -> ComputeScaleForward<'a> {
        return ComputeScaleForward {
            specs,
            animation,
            optimizer,
        };
    }
}

impl<'a> JointVisitor for ComputeScaleForward<'a> {
    fn visitor(&mut self, _joint: i32, _parent: i32) {
        let parent_scale = self.specs[_parent as usize].scale;
        let joint_scale = self.specs[_joint as usize].scale;

        let joint_spec = &mut self.specs[_joint as usize];

        // Compute joint maximum animated scale.
        let mut max_scale = 0.0;
        let track = &self.animation.tracks[_joint as usize];
        if track.scales.len() != 0 {
            for j in 0..track.scales.len() {
                let scale = &track.scales[j].value;
                let max_element = f32::max(
                    f32::max(f32::abs(scale.x), f32::abs(scale.y)), f32::abs(scale.z));
                max_scale = f32::max(max_scale, max_element);
            }
        } else {
            max_scale = 1.0;  // Default scale.
        }

        // Accumulate with parent scale.
        joint_spec.scale = max_scale;
        if _parent != crate::skeleton::Constants::KNoParent as i32 {
            joint_spec.scale *= parent_scale;
        }

        // Computes self setting distance and tolerance.
        // distance is now scaled with accumulated parent scale.
        let setting = get_joint_setting(self.optimizer, _joint);
        joint_spec.length = setting.distance * joint_scale;
        joint_spec.tolerance = setting.tolerance;
    }
}

//--------------------------------------------------------------------------------------------------
// Propagate child translations back to the root.
struct ComputeLengthBackward<'a> {
    // Defines the length of a joint hierarchy (of all child).
    specs: &'a mut Vec<Spec>,

    // Targeted animation.
    animation: &'a RawAnimation,

    // Useful to access settings and compute hierarchy length.
    optimizer: &'a AnimationOptimizer,
}

impl<'a> ComputeLengthBackward<'a> {
    fn new(specs: &'a mut Vec<Spec>, animation: &'a RawAnimation,
           optimizer: &'a AnimationOptimizer) -> ComputeLengthBackward<'a> {
        return ComputeLengthBackward {
            specs,
            animation,
            optimizer,
        };
    }
}

impl<'a> JointVisitor for ComputeLengthBackward<'a> {
    fn visitor(&mut self, _joint: i32, _parent: i32) {
        // Self translation doesn't matter if joint has no parent.
        if _parent == crate::skeleton::Constants::KNoParent as i32 {
            return;
        }

        // Compute joint maximum animated length.
        let mut max_length_sq = 0.0;
        let track = &self.animation.tracks[_joint as usize];
        for j in 0..track.translations.len() {
            max_length_sq = f32::max(max_length_sq,
                                     track.translations[j].value.length_sqr());
        }
        let max_length = f32::sqrt(max_length_sq);

        let joint_spec_length = self.specs[_joint as usize].length;
        let joint_spec_tolerance = self.specs[_joint as usize].tolerance;
        let parent_spec = &mut self.specs[_parent as usize];

        // Set parent hierarchical spec to its most impacting child, aka max
        // length and min tolerance.
        parent_spec.length = f32::max(parent_spec.length,
                                      joint_spec_length + max_length * parent_spec.scale);
        parent_spec.tolerance = f32::min(parent_spec.tolerance,
                                         joint_spec_tolerance);
    }
}

//--------------------------------------------------------------------------------------------------
struct PositionAdapter {
    scale_: f32,
}

impl PositionAdapter {
    fn new(_scale: f32) -> PositionAdapter {
        return PositionAdapter {
            scale_: _scale
        };
    }

    fn decimable(&self, _: &TranslationKey) -> bool {
        return true;
    }

    fn lerp(&self, _left: &TranslationKey, _right: &TranslationKey,
            _ref: &TranslationKey) -> TranslationKey {
        let alpha = (_ref.time - _left.time) / (_right.time - _left.time);
        debug_assert!(alpha >= 0.0 && alpha <= 1.0);
        let key = TranslationKey {
            time: _ref.time,
            value: lerp_translation(&_left.value, &_right.value, alpha),
        };
        return key;
    }

    fn distance(&self, _a: &TranslationKey, _b: &TranslationKey) -> f32 {
        return (_a.value - _b.value).length() * self.scale_;
    }
}

//--------------------------------------------------------------------------------------------------
struct RotationAdapter {
    radius_: f32,
}

impl RotationAdapter {
    fn new(_radius: f32) -> RotationAdapter {
        return RotationAdapter {
            radius_: _radius
        };
    }

    fn decimable(&self, _: &RotationKey) -> bool {
        return true;
    }

    fn lerp(&self, _left: &RotationKey, _right: &RotationKey,
            _ref: &RotationKey) -> RotationKey {
        let alpha = (_ref.time - _left.time) / (_right.time - _left.time);
        debug_assert!(alpha >= 0.0 && alpha <= 1.0);
        let key = RotationKey {
            time: _ref.time,
            value: lerp_rotation(&_left.value, &_right.value, alpha),
        };
        return key;
    }

    fn distance(&self, _left: &RotationKey, _right: &RotationKey) -> f32 {
        // Compute the shortest unsigned angle between the 2 quaternions.
        // cos_half_angle is w component of a-1 * b.
        let cos_half_angle = _left.value.dot(&_right.value);
        let sine_half_angle =
            f32::sqrt(1.0 - f32::min(1.0, cos_half_angle * cos_half_angle));
        // Deduces distance between 2 points on a circle with radius and a given
        // angle. Using half angle helps as it allows to have a right-angle
        // triangle.
        let distance = 2.0 * sine_half_angle * self.radius_;
        return distance;
    }
}

//--------------------------------------------------------------------------------------------------
struct ScaleAdapter {
    length_: f32,
}

impl ScaleAdapter {
    fn new(_length: f32) -> ScaleAdapter {
        return ScaleAdapter {
            length_: _length
        };
    }

    fn decimable(&self, _: &ScaleKey) -> bool {
        return true;
    }

    fn lerp(&self, _left: &ScaleKey, _right: &ScaleKey,
            _ref: &ScaleKey) -> ScaleKey {
        let alpha = (_ref.time - _left.time) / (_right.time - _left.time);
        debug_assert!(alpha >= 0.0 && alpha <= 1.0);
        let key = ScaleKey {
            time: _ref.time,
            value: lerp_scale(&_left.value, &_right.value, alpha),
        };
        return key;
    }

    fn distance(&self, _left: &ScaleKey, _right: &ScaleKey) -> f32 {
        return (_left.value - _right.value).length() * self.length_;
    }
}





