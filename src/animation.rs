/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use crate::animation_keyframe::*;

// Defines a runtime skeletal animation clip.
// The runtime animation data structure stores animation keyframes, for all the
// joints of a skeleton. This structure is usually filled by the
// AnimationBuilder and deserialized/loaded at runtime.
// For each transformation type (translation, rotation and scale), Animation
// structure stores a single array of keyframes that contains all the tracks
// required to animate all the joints of a skeleton, matching breadth-first
// joints order of the runtime skeleton structure. In order to optimize cache
// coherency when sampling the animation, Keyframes in this array are sorted by
// time, then by track number.
pub struct Animation {
    // Duration of the animation clip.
    duration_: f32,

    // The number of joint tracks. Can differ from the data stored in translation/
    // rotation/scale buffers because of SoA requirements.
    num_tracks_: i32,

    // Animation name.
    name_: Option<String>,

    // Stores all translation/rotation/scale keys begin and end of buffers.
    translations_: Vec<Float3Key>,
    rotations_: Vec<QuaternionKey>,
    scales_: Vec<Float3Key>,
}

impl Animation {
    // Builds a default animation.
    pub fn new() -> Animation {
        return Animation {
            duration_: 0.0,
            num_tracks_: 0,
            name_: None,
            translations_: vec![],
            rotations_: vec![],
            scales_: vec![],
        };
    }

    // Gets the animation clip duration.
    pub fn duration(&self) -> f32 { return self.duration_; }

    // Gets the number of animated tracks.
    pub fn num_tracks(&self) -> i32 { return self.num_tracks_; }

    // Returns the number of SoA elements matching the number of tracks of *this
    // animation. This value is useful to allocate SoA runtime data structures.
    pub fn num_soa_tracks(&self) -> i32 { return (self.num_tracks_ + 3) / 4; }

    // Gets animation name.
    pub fn name(&self) -> String {
        return self.name_.as_ref().unwrap_or(&"".to_string()).clone();
    }

    // Gets the buffer of translations keys.
    pub fn translations(&self) -> &Vec<Float3Key> {
        return &self.translations_;
    }

    // Gets the buffer of rotation keys.
    pub fn rotations(&self) -> &Vec<QuaternionKey> { return &self.rotations_; }

    // Gets the buffer of scale keys.
    pub fn scales(&self) -> &Vec<Float3Key> { return &self.scales_; }

    // Internal destruction function.
    pub fn allocate(&mut self, _translation_count: usize,
                    _rotation_count: usize, _scale_count: usize) {
        debug_assert!(self.name_.is_none() && self.translations_.len() == 0 &&
            self.rotations_.len() == 0 && self.scales_.len() == 0);

        // Fix up pointers. Serves larger alignment values first.
        self.translations_.resize(_translation_count, Float3Key::new());
        self.rotations_.resize(_rotation_count, QuaternionKey::new());
        self.scales_.resize(_scale_count, Float3Key::new());
    }
}