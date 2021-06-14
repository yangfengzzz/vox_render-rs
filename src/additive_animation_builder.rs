/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use crate::raw_animation::{RawAnimation, KeyType, JointTrack};
use crate::transform::Transform;
use crate::vec_float::Float3;
use crate::quaternion::Quaternion;
use crate::raw_animation;

// Defines the class responsible for building a delta animation from an offline
// raw animation. This is used to create animations compatible with additive
// blending.
pub struct AdditiveAnimationBuilder {}

impl AdditiveAnimationBuilder {
    // Builds delta animation from _input..
    // Returns true on success and fills _output_animation with the delta
    // version of _input animation.
    // *_output must be a valid RawAnimation instance. Uses first frame as
    // reference pose Returns false on failure and resets _output to an empty
    // animation. See RawAnimation::Validate() for more details about failure
    // reasons.
    pub fn apply(_input: &RawAnimation, _output: &mut RawAnimation) -> bool {
        // Reset output animation to default.
        *_output = RawAnimation::new();

        // Validate animation.
        if !_input.validate() {
            return false;
        }

        // Rebuilds output animation.
        _output.duration = _input.duration;
        _output.tracks.resize(_input.tracks.len(), JointTrack::new());

        for i in 0.._input.tracks.len() {
            let track_in = &_input.tracks[i];
            let track_out = &mut _output.tracks[i];

            let translations = &track_in.translations;
            let ref_translation = match translations.len() > 0 {
                true => translations[0].value,
                false => Float3::zero()
            };

            let rotations = &track_in.rotations;
            let ref_rotation = match rotations.len() > 0 {
                true => rotations[0].value,
                false => Quaternion::identity(),
            };

            let scales = &track_in.scales;
            let ref_scale = match scales.len() > 0 {
                true => scales[0].value,
                false => Float3::one(),
            };

            make_delta(translations, &ref_translation, make_delta_translation,
                       &mut track_out.translations);
            make_delta(rotations, &ref_rotation, make_delta_rotation,
                       &mut track_out.rotations);
            make_delta(scales, &ref_scale, make_delta_scale,
                       &mut track_out.scales);
        }

        // Output animation is always valid though.
        return _output.validate();
    }

    // Builds delta animation from _input..
    // Returns true on success and fills _output_animation with the delta
    // *_output must be a valid RawAnimation instance.
    // version of _input animation.
    // *_reference_pose used as the base pose to calculate deltas from
    // Returns false on failure and resets _output to an empty animation.
    pub fn apply_pos(_input: &RawAnimation, _reference_pose: &Vec<Transform>,
                     _output: &mut RawAnimation) -> bool {
        // Reset output animation to default.
        *_output = RawAnimation::new();

        // Validate animation.
        if !_input.validate() {
            return false;
        }

        // The reference pose must have at least the same number of
        // tracks as the raw animation.
        if _input.num_tracks() > _reference_pose.len() as i32 {
            return false;
        }

        // Rebuilds output animation.
        _output.duration = _input.duration;
        _output.tracks.resize(_input.tracks.len(), JointTrack::new());

        for i in 0.._input.tracks.len() {
            make_delta(&_input.tracks[i].translations, &_reference_pose[i].translation,
                       make_delta_translation, &mut _output.tracks[i].translations);
            make_delta(&_input.tracks[i].rotations, &_reference_pose[i].rotation,
                       make_delta_rotation, &mut _output.tracks[i].rotations);
            make_delta(&_input.tracks[i].scales, &_reference_pose[i].scale,
                       make_delta_scale, &mut _output.tracks[i].scales);
        }

        // Output animation is always valid though.
        return _output.validate();
    }
}

fn make_delta_translation(_reference: &Float3, _value: &Float3) -> Float3 {
    return _value - _reference;
}

fn make_delta_rotation(_reference: &Quaternion, _value: &Quaternion) -> Quaternion {
    return *_value * _reference.conjugate();
}

fn make_delta_scale(_reference: &Float3, _value: &Float3) -> Float3 {
    return _value / _reference;
}

fn make_delta<T, _RawTrack: KeyType<T> + raw_animation::KeyType<T, ImplType=_RawTrack>>(
    _src: &Vec<_RawTrack>, reference: &T, _make_delta: fn(_reference: &T, _value: &T) -> T,
    _dest: &mut Vec<_RawTrack>) {
    _dest.reserve(_src.len());

    // Early out if no key.
    if _src.is_empty() {
        return;
    }

    // Copy animation keys.
    for i in 0.._src.len() {
        let delta = _RawTrack::new(
            _src[i].time(),
            _make_delta(reference, &_src[i].value()),
        );
        _dest.push(delta);
    }
}

#[cfg(test)]
mod additive_animation_builder {
    use crate::raw_animation::{RawAnimation, JointTrack};
    use crate::additive_animation_builder::AdditiveAnimationBuilder;
    use crate::transform::Transform;

    #[test]
    fn error() {
        {  // Invalid input animation.
            let mut input = RawAnimation::new();
            input.duration = -1.0;
            assert_eq!(input.validate(), false);

            // Builds animation
            let mut output = RawAnimation::new();
            output.duration = -1.0;
            output.tracks.resize(1, JointTrack::new());
            assert_eq!(AdditiveAnimationBuilder::apply(&input, &mut output), false);
            assert_eq!(output.duration, RawAnimation::new().duration);
            assert_eq!(output.num_tracks(), 0);
        }

        {  // Invalid input animation with custom refpose
            let mut input = RawAnimation::new();
            input.duration = 1.0;
            input.tracks.resize(1, JointTrack::new());

            // Builds animation
            let mut output = RawAnimation::new();
            output.duration = -1.0;
            output.tracks.resize(1, JointTrack::new());

            let empty_ref_pose_range:Vec<Transform> = Vec::new();

            assert_eq!(AdditiveAnimationBuilder::apply_pos(&input, &empty_ref_pose_range, &mut output), false);
            assert_eq!(output.duration, RawAnimation::new().duration);
            assert_eq!(output.num_tracks(), 0);
        }
    }
}