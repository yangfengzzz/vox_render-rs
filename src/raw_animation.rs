/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use crate::vec_float::Float3;
use crate::quaternion::Quaternion;

// Defines a raw translation key frame.
pub struct TranslationKey {
    // Key frame time.
    pub time: f32,

    // Key frame value.
    pub value: Float3,
}

impl TranslationKey {
    // Provides identity transformation for a translation key.
    pub fn identity() -> Float3 { return Float3::zero(); }
}

//--------------------------------------------------------------------------------------------------
// Defines a raw rotation key frame.
pub struct RotationKey {
    // Key frame time.
    pub time: f32,

    // Key frame value.
    pub value: Quaternion,
}

impl RotationKey {
    // Provides identity transformation for a rotation key.
    pub fn identity() -> Quaternion { return Quaternion::identity(); }
}

//--------------------------------------------------------------------------------------------------
// Defines a raw scaling key frame.
pub struct ScaleKey {
    // Key frame time.
    pub time: f32,

    // Key frame value.
    value: Float3,
}

impl ScaleKey {
    // Provides identity transformation for a scale key.
    pub fn identity() -> Float3 { return Float3::one(); }
}

//--------------------------------------------------------------------------------------------------
// Defines a track of key frames for a bone, including translation, rotation
// and scale.
pub struct JointTrack {
    pub translations: Vec<TranslationKey>,
    pub rotations: Vec<RotationKey>,
    pub scales: Vec<ScaleKey>,
}

impl JointTrack {
    // Validates track. See RawAnimation::validate for more details.
    // Use an infinite value for _duration if unknown. This will validate
    // keyframe orders, but not maximum duration.
    pub fn validate(&self, _duration: f32) -> bool {
        todo!()
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

