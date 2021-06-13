/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use crate::raw_animation::*;
use crate::animation::Animation;
use crate::vec_float::Float3;

// Defines the class responsible of building runtime animation instances from
// offline raw animations.
// No optimization at all is performed on the raw animation.
struct AnimationBuilder {}

impl AnimationBuilder {
    // Creates an Animation based on _raw_animation and *this builder parameters.
    // Returns a valid Animation on success.
    // See RawAnimation::Validate() for more details about failure reasons.
    // The animation is returned as an unique_ptr as ownership is given back to
    // the caller.
    pub fn apply(_raw_animation: &RawAnimation) -> Animation {
        todo!()
    }
}

//--------------------------------------------------------------------------------------------------
trait SortingType {
    fn new(track: u16, prev_key_time: f32, time: f32) -> Self;

    fn track(&self) -> u16;

    fn time(&self) -> f32;
}

struct SortingTranslationKey {
    track: u16,
    prev_key_time: f32,
    key: TranslationKey,
}

impl SortingType for SortingTranslationKey {
    fn new(track: u16, prev_key_time: f32, time: f32) -> Self {
        return SortingTranslationKey {
            track,
            prev_key_time,
            key: TranslationKey { time, value: TranslationKey::identity() },
        };
    }

    fn track(&self) -> u16 {
        return self.track;
    }

    fn time(&self) -> f32 {
        return self.key.time;
    }
}

struct SortingRotationKey {
    track: u16,
    prev_key_time: f32,
    key: RotationKey,
}

impl SortingType for SortingRotationKey {
    fn new(track: u16, prev_key_time: f32, time: f32) -> Self {
        return SortingRotationKey {
            track,
            prev_key_time,
            key: RotationKey { time, value: RotationKey::identity() },
        };
    }

    fn track(&self) -> u16 {
        return self.track;
    }

    fn time(&self) -> f32 {
        return self.key.time;
    }
}

struct SortingScaleKey {
    track: u16,
    prev_key_time: f32,
    key: ScaleKey,
}

impl SortingType for SortingScaleKey {
    fn new(track: u16, prev_key_time: f32, time: f32) -> Self {
        return SortingScaleKey {
            track,
            prev_key_time,
            key: ScaleKey { time, value: ScaleKey::identity() },
        };
    }

    fn track(&self) -> u16 {
        return self.track;
    }

    fn time(&self) -> f32 {
        return self.key.time;
    }
}

fn push_back_identity_key<_DestKey: SortingType>(_track: u16, _time: f32, _dest: &mut Vec<_DestKey>) {
    let mut prev_time = -1.0;
    if !_dest.is_empty() && _dest.last().unwrap().track() == _track {
        prev_time = _dest.last().unwrap().time();
    }
    let key = _DestKey::new(_track, prev_time, _time);
    _dest.push(key);
}

// Copies a track from a RawAnimation to an Animation.
// Also fixes up the front (t = 0) and back keys (t = duration).
fn copy_raw<_SrcKey: KeyType, _DestKey: SortingType>(_src: &Vec<_SrcKey>, _track: u16, _duration: f32,
                                                     _dest: &mut Vec<_DestKey>) {
    todo!()
}












