/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use crate::animation::Animation;
use crate::animation_keyframe::KeyframeType;

// Count translation, rotation or scale keyframes for a given track number. Use
// a negative _track value to count all tracks.
pub fn count_translation_keyframes(_animation: &Animation, _track: Option<i32>) -> i32 {
    return count_keyframes_impl(_animation.translations(), _track.unwrap_or(-1));
}

pub fn count_rotation_keyframes(_animation: &Animation, _track: Option<i32>) -> i32 {
    return count_keyframes_impl(_animation.rotations(), _track.unwrap_or(-1));
}

pub fn count_scale_keyframes(_animation: &Animation, _track: Option<i32>) -> i32 {
    return count_keyframes_impl(_animation.scales(), _track.unwrap_or(-1));
}

#[inline]
fn count_keyframes_impl<_Key: KeyframeType>(_keys: &Vec<_Key>, _track: i32) -> i32 {
    if _track < 0 {
        return _keys.len() as i32;
    }

    let mut count = 0;
    for key in _keys {
        if key.track() == _track as u16 {
            count += 1;
        }
    }
    return count;
}

//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------
#[cfg(test)]
mod animation_utils {
    use crate::animation::Animation;
    use crate::raw_animation::*;
    use crate::vec_float::Float3;
    use crate::quaternion::Quaternion;
    use crate::animation_builder::AnimationBuilder;
    use crate::animation_utils::*;

    #[test]
    fn count_keyframes() {
        // Builds a valid animation.
        let animation: Option<Animation>;

        {
            let mut raw_animation = RawAnimation::new();
            raw_animation.duration = 1.0;
            raw_animation.tracks.resize(2, JointTrack::new());

            let t_key0 = TranslationKey {
                time: 0.0,
                value: Float3::new(93.0, 58.0, 46.0),
            };
            raw_animation.tracks[0].translations.push(t_key0);
            let t_key1 = TranslationKey {
                time: 0.9,
                value: Float3::new(46.0, 58.0, 93.0),
            };
            raw_animation.tracks[0].translations.push(t_key1);
            let t_key2 = TranslationKey {
                time: 1.0,
                value: Float3::new(46.0, 58.0, 99.0),
            };
            raw_animation.tracks[0].translations.push(t_key2);

            let r_key = RotationKey {
                time:
                0.7,
                value: Quaternion::new(0.0, 1.0, 0.0, 0.0),
            };
            raw_animation.tracks[0].rotations.push(r_key);

            let s_key = ScaleKey { time: 0.1, value: Float3::new(99.0, 26.0, 14.0) };
            raw_animation.tracks[1].scales.push(s_key);

            animation = AnimationBuilder::apply(&raw_animation);
            assert_eq!(animation.is_some(), true);
        }

        // 4 more tracks than expected due to SoA
        assert_eq!(count_translation_keyframes(animation.as_ref().unwrap(), Some(-1)), 9);
        assert_eq!(count_translation_keyframes(animation.as_ref().unwrap(), Some(0)), 3);
        assert_eq!(count_translation_keyframes(animation.as_ref().unwrap(), Some(1)), 2);

        assert_eq!(count_rotation_keyframes(animation.as_ref().unwrap(), Some(-1)), 8);
        assert_eq!(count_rotation_keyframes(animation.as_ref().unwrap(), Some(0)), 2);
        assert_eq!(count_rotation_keyframes(animation.as_ref().unwrap(), Some(1)), 2);

        assert_eq!(count_scale_keyframes(animation.as_ref().unwrap(), Some(-1)), 8);
        assert_eq!(count_scale_keyframes(animation.as_ref().unwrap(), Some(0)), 2);
        assert_eq!(count_scale_keyframes(animation.as_ref().unwrap(), Some(1)), 2);
    }
}