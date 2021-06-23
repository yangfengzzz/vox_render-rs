/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use std::collections::BTreeMap;
use crate::raw_animation::*;
use crate::skeleton::Skeleton;
use crate::skeleton_utils::*;
use crate::raw_animation_utils::*;

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
    // See RawAnimation::validate() for more details about failure reasons.
    pub fn apply(&self, _input: &RawAnimation, _skeleton: &Skeleton, _output: &mut RawAnimation) -> bool {
        // Reset output animation to default.
        *_output = RawAnimation::new();

        // validate animation.
        if !_input.validate() {
            return false;
        }

        let num_tracks = _input.num_tracks();

        // Validates the skeleton matches the animation.
        if num_tracks != _skeleton.num_joints() {
            return false;
        }

        // First computes bone lengths, that will be used when filtering.
        let hierarchy = HierarchyBuilder::new(&_input, &_skeleton, &self);

        // Rebuilds output animation.
        _output.name = _input.name.clone();
        _output.duration = _input.duration;
        _output.tracks.resize(num_tracks as usize, JointTrack::new());

        for i in 0..num_tracks as usize {
            let input = &_input.tracks[i];
            let output = &mut _output.tracks[i];

            // Gets joint specs back.
            let joint_length = hierarchy.specs[i].length;
            let parent = _skeleton.joint_parents()[i];
            let parent_scale = match parent != crate::skeleton::Constants::KNoParent as i16 {
                true => hierarchy.specs[parent as usize].scale,
                false => 1.0
            };
            let tolerance = hierarchy.specs[i].tolerance;

            // Filters independently T, R and S tracks.
            // This joint translation is affected by parent scale.
            let tadap = PositionAdapter::new(parent_scale);
            crate::decimate::decimate(&input.translations, &tadap, tolerance, &mut output.translations);
            // This joint rotation affects children translations/length.
            let radap = RotationAdapter::new(joint_length);
            crate::decimate::decimate(&input.rotations, &radap, tolerance, &mut output.rotations);
            // This joint scale affects children translations/length.
            let sadap = ScaleAdapter::new(joint_length);
            crate::decimate::decimate(&input.scales, &sadap, tolerance, &mut output.scales);
        }

        // Output animation is always valid though.
        return _output.validate();
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
        iterate_joints_df_reverse(_skeleton, ComputeLengthBackward::new(&mut builder.specs, &builder.animation));

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
        let parent_scale = if _parent != crate::skeleton::Constants::KNoParent as i32 {
            self.specs[_parent as usize].scale
        } else {
            0.0
        };
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
}

impl<'a> ComputeLengthBackward<'a> {
    fn new(specs: &'a mut Vec<Spec>, animation: &'a RawAnimation) -> ComputeLengthBackward<'a> {
        return ComputeLengthBackward {
            specs,
            animation,
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
pub trait DecimateType<Key> {
    fn decimable(&self, _: &Key) -> bool;
    fn lerp(&self, _left: &Key, _right: &Key, _ref: &Key) -> Key;
    fn distance(&self, _a: &Key, _b: &Key) -> f32;
}

struct PositionAdapter {
    scale_: f32,
}

impl PositionAdapter {
    fn new(_scale: f32) -> PositionAdapter {
        return PositionAdapter {
            scale_: _scale
        };
    }
}

impl DecimateType<TranslationKey> for PositionAdapter {
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
}

impl DecimateType<RotationKey> for RotationAdapter {
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
}

impl DecimateType<ScaleKey> for ScaleAdapter {
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

//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------
#[cfg(test)]
mod animation_optimizer {
    use crate::animation_optimizer::{AnimationOptimizer, Setting};
    use crate::raw_animation::{RawAnimation, JointTrack, TranslationKey, ScaleKey, RotationKey};
    use crate::skeleton::Skeleton;
    use crate::raw_skeleton::{RawSkeleton, Joint};
    use crate::skeleton_builder::SkeletonBuilder;
    use crate::vec_float::Float3;
    use crate::quaternion::Quaternion;

    #[test]
    fn error() {
        let optimizer = AnimationOptimizer::new();

        {  // Invalid input animation.
            let mut raw_skeleton = RawSkeleton::new();
            raw_skeleton.roots.resize(1, Joint::new());
            let skeleton = SkeletonBuilder::apply(&raw_skeleton);
            assert_eq!(skeleton.is_some(), true);

            let mut input = RawAnimation::new();
            input.duration = -1.0;
            assert_eq!(input.validate(), false);

            // Builds animation
            let mut output = RawAnimation::new();
            output.duration = -1.0;
            output.tracks.resize(1, JointTrack::new());
            assert_eq!(optimizer.apply(&input, skeleton.as_ref().unwrap(), &mut output), false);
            assert_eq!(output.duration, RawAnimation::new().duration);
            assert_eq!(output.num_tracks(), 0);
        }

        {  // Invalid skeleton.
            let skeleton = Skeleton::new();

            let mut input = RawAnimation::new();
            input.tracks.resize(1, JointTrack::new());
            assert_eq!(input.validate(), true);

            // Builds animation
            let mut output = RawAnimation::new();
            assert_eq!(optimizer.apply(&input, &skeleton, &mut output), false);
            assert_eq!(output.duration, RawAnimation::new().duration);
            assert_eq!(output.num_tracks(), 0);
        }
    }

    #[test]
    fn name() {
        // Prepares a skeleton.
        let raw_skeleton = RawSkeleton::new();
        let skeleton = SkeletonBuilder::apply(&raw_skeleton);
        assert_eq!(skeleton.is_some(), true);

        let optimizer = AnimationOptimizer::new();

        let mut input = RawAnimation::new();
        input.name = "Test_Animation".to_string();
        input.duration = 1.0;

        assert_eq!(input.validate(), true);

        let mut output = RawAnimation::new();
        assert_eq!(optimizer.apply(&input, skeleton.as_ref().unwrap(), &mut output), true);
        assert_eq!(output.num_tracks(), 0);
        assert_eq!(output.name, "Test_Animation");
    }

    #[test]
    fn optimize() {
        // Prepares a skeleton.
        let mut raw_skeleton = RawSkeleton::new();
        raw_skeleton.roots.resize(1, Joint::new());
        raw_skeleton.roots[0].children.resize(1, Joint::new());
        raw_skeleton.roots[0].children[0].children.resize(1, Joint::new());
        raw_skeleton.roots[0].children[0].children[0].children.resize(2, Joint::new());
        let skeleton = SkeletonBuilder::apply(&raw_skeleton);
        assert_eq!(skeleton.is_some(), true);

        // Disable non hierarchical optimizations
        let mut optimizer = AnimationOptimizer::new();

        // Disables vertex distance.
        optimizer.setting.distance = 0.0;

        let mut input = RawAnimation::new();
        input.duration = 1.0;
        input.tracks.resize(5, JointTrack::new());

        // Translations on track 0.
        {
            let key = TranslationKey { time: 0.0, value: Float3::new(4.0, 0.0, 0.0) };
            input.tracks[0].translations.push(key);
        }

        // Translations on track 1.
        {
            let key = TranslationKey { time: 0.0, value: Float3::new(0.0, 0.0, 0.0) };
            input.tracks[1].translations.push(key);
        }

        // Translations on track 2.
        {
            let key = TranslationKey { time: 0.0, value: Float3::new(5.0, 0.0, 0.0) };
            input.tracks[2].translations.push(key);
        }
        {
            let key = TranslationKey { time: 0.1, value: Float3::new(6.0, 0.0, 0.0) };
            input.tracks[2].translations.push(key);
        }
        {  // Creates an variation.
            let key = TranslationKey { time: 0.2, value: Float3::new(7.1, 0.0, 0.0) };
            input.tracks[2].translations.push(key);
        }
        {
            let key = TranslationKey { time: 0.3, value: Float3::new(8.0, 0.0, 0.0) };
            input.tracks[2].translations.push(key);
        }

        // Translations on track 3.
        {
            let key = TranslationKey { time: 0.0, value: Float3::new(16.0, 0.0, 0.0) };
            input.tracks[3].translations.push(key);
        }
        // Translations on track 4.
        {
            let key = TranslationKey { time: 0.0, value: Float3::new(32.0, 0.0, 0.0) };
            input.tracks[4].translations.push(key);
        }

        assert_eq!(input.validate(), true);

        // Small translation tolerance -> all key maintained
        {
            let mut output = RawAnimation::new();
            optimizer.setting.tolerance = 0.01;
            assert_eq!(optimizer.apply(&input, skeleton.as_ref().unwrap(), &mut output), true);
            assert_eq!(output.num_tracks(), 5);

            let translations = &output.tracks[2].translations;
            assert_eq!(translations.len(), 4);
            assert_eq!(translations[0].time, 0.0);
            assert_eq!(translations[1].time, 0.1);
            assert_eq!(translations[2].time, 0.2);
            assert_eq!(translations[3].time, 0.3);
        }

        // High translation tolerance -> all key interpolated
        {
            let mut output = RawAnimation::new();
            optimizer.setting.tolerance = 0.1;
            assert_eq!(optimizer.apply(&input, skeleton.as_ref().unwrap(), &mut output), true);
            assert_eq!(output.num_tracks(), 5);

            let translations = &output.tracks[2].translations;
            assert_eq!(translations.len(), 2);
            assert_eq!(translations[0].time, 0.0);
            assert_eq!(translations[1].time, 0.3);
        }

        // Introduces a 10x scaling upstream that amplifies error
        // Scaling on track 0
        {
            let key = ScaleKey { time: 0.0, value: Float3::new(10.0, 0.0, 0.0) };
            input.tracks[0].scales.push(key);
        }

        // High translation tolerance -> keys aren't interpolated because of scale
        // effect.
        {
            let mut output = RawAnimation::new();
            optimizer.setting.tolerance = 0.1;
            assert_eq!(optimizer.apply(&input, skeleton.as_ref().unwrap(), &mut output), true);
            assert_eq!(output.num_tracks(), 5);

            let translations = &output.tracks[2].translations;
            assert_eq!(translations.len(), 4);
        }

        // Very high tolerance
        {
            let mut output = RawAnimation::new();
            optimizer.setting.tolerance = 1.0;
            assert_eq!(optimizer.apply(&input, skeleton.as_ref().unwrap(), &mut output), true);
            assert_eq!(output.num_tracks(), 5);

            let translations = &output.tracks[2].translations;
            assert_eq!(translations.len(), 2);
        }

        // Introduces a -10x scaling upstream that amplifies error
        // Scaling on track 0
        { input.tracks[0].scales[0].value = Float3::new(0.0, -10.0, 0.0); }

        // High translation tolerance -> keys aren't interpolated because of scale
        // effect.
        {
            let mut output = RawAnimation::new();
            optimizer.setting.tolerance = 0.1;
            assert_eq!(optimizer.apply(&input, skeleton.as_ref().unwrap(), &mut output), true);
            assert_eq!(output.num_tracks(), 5);

            let translations = &output.tracks[2].translations;
            assert_eq!(translations.len(), 4);
            assert_eq!(translations[0].time, 0.0);
            assert_eq!(translations[1].time, 0.1);
            assert_eq!(translations[2].time, 0.2);
            assert_eq!(translations[3].time, 0.3);
        }

        // Very high tolerance
        {
            let mut output = RawAnimation::new();
            optimizer.setting.tolerance = 1.0;
            assert_eq!(optimizer.apply(&input, skeleton.as_ref().unwrap(), &mut output), true);
            assert_eq!(output.num_tracks(), 5);

            let translations = &output.tracks[2].translations;
            assert_eq!(translations.len(), 2);
            assert_eq!(translations[0].time, 0.0);
            assert_eq!(translations[1].time, 0.3);
        }

        // Compensate scale on next joint
        {
            let key = ScaleKey { time: 0.0, value: Float3::new(0.1, 0.0, 0.0) };
            input.tracks[1].scales.push(key);
        }

        // High translation tolerance -> keys ar interpolated because of scale
        // compensation.
        {
            let mut output = RawAnimation::new();
            optimizer.setting.tolerance = 1.0;
            assert_eq!(optimizer.apply(&input, skeleton.as_ref().unwrap(), &mut output), true);
            assert_eq!(output.num_tracks(), 5);

            let translations = &output.tracks[2].translations;
            assert_eq!(translations.len(), 2);
        }

        // Remove scaling compensation
        { input.tracks[1].scales.clear(); }

        // Introduces a .1x scaling upstream that amplifies error
        // Scaling on track 0
        { input.tracks[0].scales[0].value = Float3::new(0.0, 0.0, 0.1); }

        // High translation tolerance -> keys aren't interpolated because of scale
        // effect.
        {
            let mut output = RawAnimation::new();
            optimizer.setting.tolerance = 0.001;
            assert_eq!(optimizer.apply(&input, skeleton.as_ref().unwrap(), &mut output), true);
            assert_eq!(output.num_tracks(), 5);

            let translations = &output.tracks[2].translations;
            assert_eq!(translations.len(), 4);
            assert_eq!(translations[0].time, 0.0);
            assert_eq!(translations[1].time, 0.1);
            assert_eq!(translations[2].time, 0.2);
            assert_eq!(translations[3].time, 0.3);
        }

        // Very high translation tolerance
        {
            let mut output = RawAnimation::new();
            optimizer.setting.tolerance = 0.01;
            assert_eq!(optimizer.apply(&input, skeleton.as_ref().unwrap(), &mut output), true);
            assert_eq!(output.num_tracks(), 5);

            let translations = &output.tracks[2].translations;
            assert_eq!(translations.len(), 2);
            assert_eq!(translations[0].time, 0.0);
            assert_eq!(translations[1].time, 0.3);
        }

        // Remove scaling
        { input.tracks[0].scales.clear(); }

        // Rotations on track 0.
        {
            let key = RotationKey {
                time: 0.0,
                value: Quaternion::from_euler(0.0, 0.0, 0.0),
            };
            input.tracks[0].rotations.push(key);
        }
        {                                     // Include error
            let angle_error = 2.5e-3;  // creates an arc of .1m at 40m.
            let key = RotationKey {
                time: 0.1,
                value: Quaternion::from_euler(crate::math_constant::K_PI_4 + angle_error,
                                              0.0, 0.0),
            };
            input.tracks[0].rotations.push(key);
        }
        {
            let key = RotationKey {
                time: 0.2,
                value: Quaternion::from_euler(crate::math_constant::K_PI_2, 0.0, 0.0),
            };
            input.tracks[0].rotations.push(key);
        }

        // Big enough tolerance -> keys rejected
        {
            let mut output = RawAnimation::new();
            optimizer.setting.tolerance = 0.3;
            optimizer.setting.distance = 40.0;
            assert_eq!(optimizer.apply(&input, skeleton.as_ref().unwrap(), &mut output), true);
            assert_eq!(output.num_tracks(), 5);

            let rotations = &output.tracks[0].rotations;
            assert_eq!(rotations.len(), 2);

            let translations = &output.tracks[2].translations;
            assert_eq!(translations.len(), 2);
        }

        // Small enough tolerance -> keys rejected
        {
            let mut output = RawAnimation::new();
            optimizer.setting.tolerance = 0.05;
            optimizer.setting.distance = 40.0;
            assert_eq!(optimizer.apply(&input, skeleton.as_ref().unwrap(), &mut output), true);
            assert_eq!(output.num_tracks(), 5);

            let rotations = &output.tracks[0].rotations;
            assert_eq!(rotations.len(), 3);

            let translations = &output.tracks[2].translations;
            assert_eq!(translations.len(), 4);
        }

        // Back to default
        optimizer.setting = Setting::new_default();

        // Small translation tolerance -> all key maintained
        {
            let mut output = RawAnimation::new();
            optimizer.setting.tolerance = 0.01;
            assert_eq!(optimizer.apply(&input, skeleton.as_ref().unwrap(), &mut output), true);
            assert_eq!(output.num_tracks(), 5);

            let rotations = &output.tracks[0].rotations;
            assert_eq!(rotations.len(), 3);

            let translations = &output.tracks[2].translations;
            assert_eq!(translations.len(), 4);
        }

        // Introduces a .1x scaling upstream that lowers error
        // Scaling on track 0
        {
            let key = ScaleKey { time: 0.0, value: Float3::new(0.0, 0.1, 0.0) };
            input.tracks[1].scales.push(key);
        }

        // Small translation tolerance, but scaled down -> keys rejected
        {
            let mut output = RawAnimation::new();
            optimizer.setting.tolerance = 0.011;
            assert_eq!(optimizer.apply(&input, skeleton.as_ref().unwrap(), &mut output), true);
            assert_eq!(output.num_tracks(), 5);

            let rotations = &output.tracks[0].rotations;
            assert_eq!(rotations.len(), 2);

            let translations = &output.tracks[2].translations;
            assert_eq!(translations.len(), 2);
        }

        // More vertex distance -> keys are maintained (translation unaffected)
        {
            let mut output = RawAnimation::new();
            optimizer.setting.tolerance = 0.01;
            optimizer.setting.distance = 1.0;
            assert_eq!(optimizer.apply(&input, skeleton.as_ref().unwrap(), &mut output), true);
            assert_eq!(output.num_tracks(), 5);

            let rotations = &output.tracks[0].rotations;
            assert_eq!(rotations.len(), 3);

            let translations = &output.tracks[2].translations;
            assert_eq!(translations.len(), 2);
        }

        // Remove scaling
        { input.tracks[2].scales.clear(); }
    }

    #[test]
    fn optimize_override() {
        // Prepares a skeleton.
        let mut raw_skeleton = RawSkeleton::new();
        raw_skeleton.roots.resize(1, Joint::new());
        raw_skeleton.roots[0].children.resize(1, Joint::new());
        raw_skeleton.roots[0].children[0].children.resize(1, Joint::new());
        raw_skeleton.roots[0].children[0].children[0].children.resize(2, Joint::new());
        let skeleton = SkeletonBuilder::apply(&raw_skeleton);
        assert_eq!(skeleton.is_some(), true);

        // Disable non hierarchical optimizations
        let mut optimizer = AnimationOptimizer::new();
        let loose_setting = Setting::new(1e-2,   // 1cm
                                         1e-3);  // 1mm
        // Disables vertex distance.
        optimizer.setting.distance = 0.0;

        let mut input = RawAnimation::new();
        input.duration = 1.0;
        input.tracks.resize(5, JointTrack::new());

        // Translations on track 0.
        {
            let key = TranslationKey { time: 0.0, value: Float3::new(0.4, 0.0, 0.0) };
            input.tracks[0].translations.push(key);
        }

        // Rotations on track 0.
        {
            let key = RotationKey {
                time: 0.0,
                value: Quaternion::from_euler(0.0, 0.0, 0.0),
            };
            input.tracks[1].rotations.push(key);
        }
        {                                   // Includes an error that
            let angle_error = 1e-3;  // creates an arc of 1mm at 1m.
            let key = RotationKey {
                time: 0.1,
                value: Quaternion::from_euler(crate::math_constant::K_PI_4 + angle_error,
                                              0.0, 0.0),
            };
            input.tracks[1].rotations.push(key);
        }
        {
            let key = RotationKey {
                time: 0.2,
                value: Quaternion::from_euler(crate::math_constant::K_PI_2, 0.0, 0.0),
            };
            input.tracks[1].rotations.push(key);
        }

        // Translations on track 1.
        {
            let key = TranslationKey { time: 0.0, value: Float3::new(0.0, 0.0, 0.0) };
            input.tracks[1].translations.push(key);
        }

        // Translations on track 2.
        {
            let key = TranslationKey { time: 0.0, value: Float3::new(0.05, 0.0, 0.0) };
            input.tracks[2].translations.push(key);
        }
        {
            let key = TranslationKey { time: 0.1, value: Float3::new(0.06, 0.0, 0.0) };
            input.tracks[2].translations.push(key);
        }
        {  // Creates a variation.
            let trans_err = 5e-4;
            let key = TranslationKey { time: 0.2, value: Float3::new(0.07 + trans_err, 0.0, 0.0) };
            input.tracks[2].translations.push(key);
        }
        {
            let key = TranslationKey { time: 0.3, value: Float3::new(0.08, 0.0, 0.0) };
            input.tracks[2].translations.push(key);
        }

        // Translations on track 3.
        {
            let key = TranslationKey { time: 0.0, value: Float3::new(0.16, 0.0, 0.0) };
            input.tracks[3].translations.push(key);
        }
        // Translations on track 4.
        {
            let key = TranslationKey { time: 0.0, value: Float3::new(0.32, 0.0, 0.0) };
            input.tracks[4].translations.push(key);
        }

        assert_eq!(input.validate(), true);

        // Default global tolerances
        {
            let mut output = RawAnimation::new();
            optimizer.setting = loose_setting.clone();
            assert_eq!(optimizer.apply(&input, skeleton.as_ref().unwrap(), &mut output), true);
            assert_eq!(output.num_tracks(), 5);

            let rotations = &output.tracks[1].rotations;
            assert_eq!(rotations.len(), 2);
            assert_eq!(rotations[0].time, 0.0);
            assert_eq!(rotations[1].time, 0.2);

            let translations = &output.tracks[2].translations;
            assert_eq!(translations.len(), 2);
            assert_eq!(translations[0].time, 0.0);
            assert_eq!(translations[1].time, 0.3);
        }

        // Overriding root has no effect on its child, even with small tolerance.
        {
            let mut output = RawAnimation::new();
            optimizer.setting = loose_setting.clone();
            let joint_override = Setting::new(1e-6, 1e6);
            optimizer.joints_setting_override.insert(0, joint_override);
            assert_eq!(optimizer.apply(&input, skeleton.as_ref().unwrap(), &mut output), true);
            assert_eq!(output.num_tracks(), 5);

            let rotations = &output.tracks[1].rotations;
            assert_eq!(rotations.len(), 2);

            let translations = &output.tracks[2].translations;
            assert_eq!(translations.len(), 2);

            optimizer.joints_setting_override.clear();
        }

        // Overriding a joint has effect on itself.
        {
            let mut output = RawAnimation::new();
            optimizer.setting = loose_setting.clone();
            let joint_override = Setting::new(1e-3,  // 1mm
                                              1e-2);  // 1cm
            optimizer.joints_setting_override.insert(1, joint_override);
            assert_eq!(optimizer.apply(&input, skeleton.as_ref().unwrap(), &mut output), true);
            assert_eq!(output.num_tracks(), 5);

            let rotations = &output.tracks[1].rotations;
            assert_eq!(rotations.len(), 2);

            let translations = &output.tracks[2].translations;
            assert_eq!(translations.len(), 2);

            optimizer.joints_setting_override.clear();
        }

        // Overriding leaf has effect up to the root though.
        {
            let mut output = RawAnimation::new();
            optimizer.setting = loose_setting.clone();
            let joint_override = Setting::new(1e-3,  // 1mm
                                              10.0);   // 10m
            optimizer.joints_setting_override.insert(2, joint_override);
            assert_eq!(optimizer.apply(&input, skeleton.as_ref().unwrap(), &mut output), true);
            assert_eq!(output.num_tracks(), 5);

            let rotations = &output.tracks[1].rotations;
            assert_eq!(rotations.len(), 3);

            let translations = &output.tracks[2].translations;
            assert_eq!(translations.len(), 2);

            optimizer.joints_setting_override.clear();
        }

        // Scale at root affects rotation and translation.
        {
            let key = ScaleKey { time: 0.0, value: Float3::new(0.1, 2.0, 0.1) };
            input.tracks[0].scales.push(key);

            let mut output = RawAnimation::new();
            optimizer.setting = loose_setting.clone();
            let joint_override = Setting::new(1.0e-3,  // > 1mm
                                              1.0);    // 1m
            optimizer.joints_setting_override.insert(1, joint_override.clone());
            optimizer.joints_setting_override.insert(2, joint_override.clone());
            assert_eq!(optimizer.apply(&input, skeleton.as_ref().unwrap(), &mut output), true);
            assert_eq!(output.num_tracks(), 5);

            let rotations = &output.tracks[1].rotations;
            assert_eq!(rotations.len(), 3);

            let translations = &output.tracks[2].translations;
            assert_eq!(translations.len(), 3);

            optimizer.joints_setting_override.clear();
            input.tracks[0].scales.clear();
        }

        // Scale at leaf doesn't affect anything but the leaf.
        {
            let key = ScaleKey { time: 0.0, value: Float3::new(0.1, 2.0, 0.1) };
            input.tracks[4].scales.push(key);

            let mut output = RawAnimation::new();
            optimizer.setting = loose_setting.clone();
            let joint_override = Setting::new(1e-3,  // < 1mm
                                              0.5);   // .5m
            optimizer.joints_setting_override.insert(1, joint_override);
            assert_eq!(optimizer.apply(&input, skeleton.as_ref().unwrap(), &mut output), true);
            assert_eq!(output.num_tracks(), 5);

            let rotations = &output.tracks[1].rotations;
            assert_eq!(rotations.len(), 2);

            let translations = &output.tracks[2].translations;
            assert_eq!(translations.len(), 2);

            optimizer.joints_setting_override.clear();
            input.tracks[4].scales.clear();
        }
    }
}