/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use crate::vec_float::Float3;
use crate::quaternion::Quaternion;

pub trait KeyType<T> {
    type ImplType;
    fn new(time: f32, value: T) -> Self::ImplType;

    fn time(&self) -> f32;

    fn value(&self) -> T;

    fn set_value(&mut self, value: T);

    fn identity() -> T;
}

// Implements key frames' time range and ordering checks.
// See AnimationBuilder::Create for more details.
fn validate_track<T, _Key: KeyType<T>>(_track: &Vec<_Key>,
                                       _duration: f32) -> bool {
    let mut previous_time = -1.0;
    for k in 0.._track.len() {
        let frame_time = _track[k].time();
        // Tests frame's time is in range [0:duration].
        if frame_time < 0.0 || frame_time > _duration {
            return false;
        }
        // Tests that frames are sorted.
        if frame_time <= previous_time {
            return false;
        }
        previous_time = frame_time;
    }
    return true;  // Validated.
}

//--------------------------------------------------------------------------------------------------
// Defines a raw translation key frame.
#[derive(Clone)]
pub struct TranslationKey {
    // Key frame time.
    pub time: f32,

    // Key frame value.
    pub value: Float3,
}

impl KeyType<Float3> for TranslationKey {
    type ImplType = TranslationKey;

    fn new(time: f32, value: Float3) -> Self::ImplType {
        return TranslationKey {
            time,
            value,
        };
    }

    fn time(&self) -> f32 {
        return self.time;
    }

    fn value(&self) -> Float3 {
        return self.value;
    }

    fn set_value(&mut self, value: Float3) {
        self.value = value;
    }

    // Provides identity transformation for a translation key.
    fn identity() -> Float3 { return Float3::zero(); }
}

//--------------------------------------------------------------------------------------------------
// Defines a raw rotation key frame.
#[derive(Clone)]
pub struct RotationKey {
    // Key frame time.
    pub time: f32,

    // Key frame value.
    pub value: Quaternion,
}

impl KeyType<Quaternion> for RotationKey {
    type ImplType = RotationKey;

    fn new(time: f32, value: Quaternion) -> Self::ImplType {
        return RotationKey {
            time,
            value,
        };
    }

    fn time(&self) -> f32 {
        return self.time;
    }

    fn value(&self) -> Quaternion {
        return self.value;
    }

    fn set_value(&mut self, value: Quaternion) {
        self.value = value;
    }

    // Provides identity transformation for a rotation key.
    fn identity() -> Quaternion { return Quaternion::identity(); }
}

//--------------------------------------------------------------------------------------------------
// Defines a raw scaling key frame.
#[derive(Clone)]
pub struct ScaleKey {
    // Key frame time.
    pub time: f32,

    // Key frame value.
    pub value: Float3,
}

impl KeyType<Float3> for ScaleKey {
    type ImplType = ScaleKey;

    fn new(time: f32, value: Float3) -> Self::ImplType {
        return ScaleKey {
            time,
            value,
        };
    }

    fn time(&self) -> f32 {
        return self.time;
    }

    fn value(&self) -> Float3 {
        return self.value;
    }

    fn set_value(&mut self, value: Float3) {
        self.value = value;
    }

    // Provides identity transformation for a scale key.
    fn identity() -> Float3 { return Float3::one(); }
}

//--------------------------------------------------------------------------------------------------
// Defines a track of key frames for a bone, including translation, rotation
// and scale.
#[derive(Clone)]
pub struct JointTrack {
    pub translations: Vec<TranslationKey>,
    pub rotations: Vec<RotationKey>,
    pub scales: Vec<ScaleKey>,
}

impl JointTrack {
    pub fn new() -> JointTrack {
        return JointTrack {
            translations: vec![],
            rotations: vec![],
            scales: vec![],
        };
    }

    // Validates track. See RawAnimation::validate for more details.
    // Use an infinite value for _duration if unknown. This will validate
    // keyframe orders, but not maximum duration.
    pub fn validate(&self, _duration: f32) -> bool {
        return validate_track(&self.translations, _duration) &&
            validate_track(&self.rotations, _duration) &&
            validate_track(&self.scales, _duration);
    }
}

//--------------------------------------------------------------------------------------------------
// Offline animation type.
// This animation type is not intended to be used in run time. It is used to
// define the offline animation object that can be converted to the runtime
// animation using the AnimationBuilder.
// This animation structure exposes tracks of keyframes. Keyframes are defined
// with a time and a value which can either be a translation (3 float x, y, z),
// a rotation (a quaternion) or scale coefficient (3 floats x, y, z). Tracks are
// defined as a set of three different std::vectors (translation, rotation and
// scales). Animation structure is then a vector of tracks, along with a
// duration value.
// Finally the RawAnimation structure exposes validate() function to check that
// it is valid, meaning that all the following rules are respected:
//  1. Animation duration is greater than 0.
//  2. Keyframes' time are sorted in a strict ascending order.
//  3. Keyframes' time are all within [0,animation duration] range.
// Animations that would fail this validation will fail to be converted by the
// AnimationBuilder.
pub struct RawAnimation {
    // Stores per joint JointTrack, ie: per joint animation key-frames.
    // tracks_.size() gives the number of animated joints.
    pub tracks: Vec<JointTrack>,

    // The duration of the animation. All the keys of a valid RawAnimation are in
    // the range [0,duration].
    pub duration: f32,

    // Name of the animation.
    pub name: String,
}

impl RawAnimation {
    // Constructs a valid RawAnimation with a 1s default duration.
    pub fn new() -> RawAnimation {
        return RawAnimation {
            tracks: vec![],
            duration: 1.0,
            name: "".to_string(),
        };
    }

    // Tests for *this validity.
    // Returns true if animation data (duration, tracks) is valid:
    //  1. Animation duration is greater than 0.
    //  2. Keyframes' time are sorted in a strict ascending order.
    //  3. Keyframes' time are all within [0,animation duration] range.
    pub fn validate(&self) -> bool {
        if self.duration <= 0.0 {  // Tests duration is valid.
            return false;
        }
        if self.tracks.len() > crate::skeleton::Constants::KMaxJoints as usize {  // Tests number of tracks.
            return false;
        }
        // Ensures that all key frames' time are valid, ie: in a strict ascending
        // order and within range [0:duration].
        let mut valid = true;
        let mut i = 0;
        while valid && i < self.tracks.len() {
            valid = self.tracks[i].validate(self.duration);
            i += 1;
        }
        return valid;  // *this is valid.
    }

    // Get the estimated animation's size in bytes.
    pub fn size(&self) -> usize {
        let mut size = std::mem::size_of::<RawAnimation>();

        // Accumulates keyframes size.
        let tracks_count = self.tracks.len();
        for i in 0..tracks_count {
            size += self.tracks[i].translations.len() * std::mem::size_of::<TranslationKey>();
            size += self.tracks[i].rotations.len() * std::mem::size_of::<RotationKey>();
            size += self.tracks[i].scales.len() * std::mem::size_of::<ScaleKey>();
        }

        // Accumulates tracks.
        size += tracks_count * std::mem::size_of::<JointTrack>();
        size += self.name.len();

        return size;
    }

    // Returns the number of tracks of this animation.
    pub fn num_tracks(&self) -> i32 { return self.tracks.len() as i32; }
}

