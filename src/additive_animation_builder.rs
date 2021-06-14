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
    use crate::raw_animation::*;
    use crate::additive_animation_builder::AdditiveAnimationBuilder;
    use crate::transform::Transform;
    use crate::vec_float::Float3;
    use crate::quaternion::Quaternion;
    use crate::math_test_helper::*;
    use crate::*;

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

            let empty_ref_pose_range: Vec<Transform> = Vec::new();

            assert_eq!(AdditiveAnimationBuilder::apply_pos(&input, &empty_ref_pose_range, &mut output), false);
            assert_eq!(output.duration, RawAnimation::new().duration);
            assert_eq!(output.num_tracks(), 0);
        }
    }

    #[test]
    fn build() {
        let mut input = RawAnimation::new();
        input.duration = 1.0;
        input.tracks.resize(3, JointTrack::new());

        // First track is empty
        {
            // input.tracks[0]
        }

        // 2nd track
        // 1 key at the beginning
        {
            let key = TranslationKey {
                time: 0.0,
                value: Float3::new(2.0, 3.0, 4.0),
            };
            input.tracks[1].translations.push(key);
        }
        {
            let key = RotationKey {
                time: 0.0,
                value: Quaternion::new(0.70710677, 0.0, 0.0, 0.70710677),
            };
            input.tracks[1].rotations.push(key);
        }
        {
            let key = ScaleKey { time: 0.0, value: Float3::new(5.0, 6.0, 7.0) };
            input.tracks[1].scales.push(key);
        }

        // 3rd track
        // 2 keys after the beginning
        {
            let key0 = TranslationKey {
                time: 0.5,
                value: Float3::new(2.0, 3.0, 4.0),
            };
            input.tracks[2].translations.push(key0);
            let key1 = TranslationKey {
                time: 0.7,
                value: Float3::new(20.0, 30.0, 40.0),
            };
            input.tracks[2].translations.push(key1);
        }
        {
            let key0 = RotationKey {
                time: 0.5,
                value: Quaternion::new(0.70710677, 0.0, 0.0, 0.70710677),
            };
            input.tracks[2].rotations.push(key0);
            let key1 = RotationKey {
                time: 0.7,
                value: Quaternion::new(-0.70710677, 0.0, 0.0, 0.70710677),
            };
            input.tracks[2].rotations.push(key1);
        }
        {
            let key0 = ScaleKey { time: 0.5, value: Float3::new(5.0, 6.0, 7.0) };
            input.tracks[2].scales.push(key0);
            let key1 = ScaleKey {
                time: 0.7,
                value: Float3::new(50.0, 60.0, 70.0),
            };
            input.tracks[2].scales.push(key1);
        }

        // Builds animation with very little tolerance.
        {
            let mut output = RawAnimation::new();
            assert_eq!(AdditiveAnimationBuilder::apply(&input, &mut output), true);
            assert_eq!(output.num_tracks(), 3);

            // 1st track.
            {
                assert_eq!(output.tracks[0].translations.len(), 0);
                assert_eq!(output.tracks[0].rotations.len(), 0);
                assert_eq!(output.tracks[0].scales.len(), 0);
            }

            // 2nd track.
            {
                let translations = &output.tracks[1].translations;
                assert_eq!(translations.len(), 1);
                assert_eq!(translations[0].time, 0.0);
                expect_float3_eq!(translations[0].value, 0.0, 0.0, 0.0);
                let rotations = &output.tracks[1].rotations;
                assert_eq!(rotations.len(), 1);
                assert_eq!(rotations[0].time, 0.0);
                expect_quaternion_eq!(rotations[0].value, 0.0, 0.0, 0.0, 1.0);
                let scales = &output.tracks[1].scales;
                assert_eq!(scales.len(), 1);
                assert_eq!(scales[0].time, 0.0);
                expect_float3_eq!(scales[0].value, 1.0, 1.0, 1.0);
            }

            // 3rd track.
            {
                let translations = &output.tracks[2].translations;
                assert_eq!(translations.len(), 2);
                assert_eq!(translations[0].time, 0.5);
                expect_float3_eq!(translations[0].value, 0.0, 0.0, 0.0);
                assert_eq!(translations[1].time, 0.7);
                expect_float3_eq!(translations[1].value, 18.0, 27.0, 36.0);
                let rotations = &output.tracks[2].rotations;
                assert_eq!(rotations.len(), 2);
                assert_eq!(rotations[0].time, 0.5);
                expect_quaternion_eq!(rotations[0].value, 0.0, 0.0, 0.0, 1.0);
                assert_eq!(rotations[1].time, 0.7);
                expect_quaternion_eq!(rotations[1].value, -1.0, 0.0, 0.0, 0.0);
                let scales = &output.tracks[2].scales;
                assert_eq!(scales.len(), 2);
                assert_eq!(scales[0].time, 0.5);
                expect_float3_eq!(scales[0].value, 1.0, 1.0, 1.0);
                assert_eq!(scales[1].time, 0.7);
                expect_float3_eq!(scales[1].value, 10.0, 10.0, 10.0);
            }
        }
    }

    #[test]
    fn build_ref_pose() {
        let mut input = RawAnimation::new();
        input.duration = 1.0;
        input.tracks.resize(3, JointTrack::new());

        // First track is empty
        {
            // input.tracks[0]
        }

        // 2nd track
        // 1 key at the beginning
        {
            let key = TranslationKey {
                time: 0.0,
                value: Float3::new(2.0, 3.0, 4.0),
            };
            input.tracks[1].translations.push(key);
        }
        {
            let key = RotationKey {
                time: 0.0,
                value: Quaternion::new(0.70710677, 0.0, 0.0, 0.70710677),
            };
            input.tracks[1].rotations.push(key);
        }
        {
            let key = ScaleKey { time: 0.0, value: Float3::new(5.0, 6.0, 7.0) };
            input.tracks[1].scales.push(key);
        }

        // 3rd track
        // 2 keys after the beginning
        {
            let key0 = TranslationKey {
                time: 0.5,
                value: Float3::new(2.0, 3.0, 4.0),
            };
            input.tracks[2].translations.push(key0);
            let key1 = TranslationKey {
                time: 0.7,
                value: Float3::new(20.0, 30.0, 40.0),
            };
            input.tracks[2].translations.push(key1);
        }
        {
            let key0 = RotationKey {
                time: 0.5,
                value: Quaternion::new(0.70710677, 0.0, 0.0, 0.70710677),
            };
            input.tracks[2].rotations.push(key0);
            let key1 = RotationKey {
                time: 0.7,
                value: Quaternion::new(-0.70710677, 0.0, 0.0, 0.70710677),
            };
            input.tracks[2].rotations.push(key1);
        }
        {
            let key0 = ScaleKey { time: 0.5, value: Float3::new(5.0, 6.0, 7.0) };
            input.tracks[2].scales.push(key0);
            let key1 = ScaleKey { time: 0.7, value: Float3::new(50.0, 60.0, 70.0) };
            input.tracks[2].scales.push(key1);
        }

        // Builds animation with a custom refpose & very little tolerance
        {
            let mut ref_pose = [Transform::identity(), Transform::identity(), Transform::identity()];
            ref_pose[0] = Transform::identity();
            ref_pose[1].translation = Float3::new(1.0, 1.0, 1.0);
            ref_pose[1].rotation = Quaternion::new(0.0, 0.0, 0.70710677, 0.70710677);
            ref_pose[1].scale = Float3::new(1.0, -1.0, 2.0);
            ref_pose[2].translation = input.tracks[2].translations[0].value;
            ref_pose[2].rotation = input.tracks[2].rotations[0].value;
            ref_pose[2].scale = input.tracks[2].scales[0].value;

            let mut output = RawAnimation::new();
            assert_eq!(
                AdditiveAnimationBuilder::apply_pos(&input, &ref_pose.to_vec(), &mut output), true);
            assert_eq!(output.num_tracks(), 3);

            // 1st track.
            {
                assert_eq!(output.tracks[0].translations.len(), 0);
                assert_eq!(output.tracks[0].rotations.len(), 0);
                assert_eq!(output.tracks[0].scales.len(), 0);
            }

            // 2nd track.
            {
                let translations = &output.tracks[1].translations;
                assert_eq!(translations.len(), 1);
                assert_eq!(translations[0].time, 0.0);
                expect_float3_eq!(translations[0].value, 1.0, 2.0, 3.0);
                let rotations = &output.tracks[1].rotations;
                assert_eq!(rotations.len(), 1);
                assert_eq!(rotations[0].time, 0.0);
                expect_quaternion_eq!(rotations[0].value, 0.5, 0.5, -0.5, 0.5);
                let scales = &output.tracks[1].scales;
                assert_eq!(scales.len(), 1);
                assert_eq!(scales[0].time, 0.0);
                expect_float3_eq!(scales[0].value, 5.0, -6.0, 3.5);
            }

            // 3rd track.
            {
                let translations = &output.tracks[2].translations;
                assert_eq!(translations.len(), 2);
                assert_eq!(translations[0].time, 0.5);
                expect_float3_eq!(translations[0].value, 0.0, 0.0, 0.0);
                assert_eq!(translations[1].time, 0.7);
                expect_float3_eq!(translations[1].value, 18.0, 27.0, 36.0);
                let rotations = &output.tracks[2].rotations;
                assert_eq!(rotations.len(), 2);
                assert_eq!(rotations[0].time, 0.5);
                expect_quaternion_eq!(rotations[0].value, 0.0, 0.0, 0.0, 1.0);
                assert_eq!(rotations[1].time, 0.7);
                expect_quaternion_eq!(rotations[1].value, -1.0, 0.0, 0.0, 0.0);
                let scales = &output.tracks[2].scales;
                assert_eq!(scales.len(), 2);
                assert_eq!(scales[0].time, 0.5);
                expect_float3_eq!(scales[0].value, 1.0, 1.0, 1.0);
                assert_eq!(scales[1].time, 0.7);
                expect_float3_eq!(scales[1].value, 10.0, 10.0, 10.0);
            }
        }
    }
}