/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

// Defines a rectangle by the integer coordinates of its lower-left and
// width-height.
pub struct RectInt {
    // Specifies the x-coordinate of the lower side.
    pub left: i32,
    // Specifies the x-coordinate of the left side.
    pub bottom: i32,
    // Specifies the width of the rectangle.
    pub width: i32,
    // Specifies the height of the rectangle..
    pub height: i32,
}

impl RectInt {
    // Constructs a uninitialized rectangle.
    pub fn new_default() -> RectInt {
        return RectInt {
            left: 0,
            bottom: 0,
            width: 0,
            height: 0,
        };
    }

    // Constructs a rectangle with the specified arguments.
    pub fn new(_left: i32, _bottom: i32, _width: i32, _height: i32) -> RectInt {
        return RectInt {
            left: _left,
            bottom: _bottom,
            width: _width,
            height: _height,
        };
    }

    // Tests whether _x and _y coordinates are within rectangle bounds.
    pub fn is_inside(&self, _x: i32, _y: i32) -> bool {
        return _x >= self.left && _x < self.left + self.width && _y >= self.bottom &&
            _y < self.bottom + self.height;
    }

    // Gets the rectangle x coordinate of the right rectangle side.
    pub fn right(&self) -> i32 { return self.left + self.width; }

    // Gets the rectangle y coordinate of the top rectangle side.
    pub fn top(&self) -> i32 { return self.bottom + self.height; }
}

//--------------------------------------------------------------------------------------------------
// Defines a rectangle by the floating point coordinates of its lower-left
// and width-height.
pub struct RectFloat {
    // Specifies the x-coordinate of the lower side.
    pub left: f32,
    // Specifies the x-coordinate of the left side.
    pub bottom: f32,
    // Specifies the width of the rectangle.
    pub width: f32,
    // Specifies the height of the rectangle.
    pub height: f32,
}

impl RectFloat {
    // Constructs a uninitialized rectangle.
    pub fn new_default() -> RectFloat {
        return RectFloat {
            left: 0.0,
            bottom: 0.0,
            width: 0.0,
            height: 0.0,
        };
    }

    // Constructs a rectangle with the specified arguments.
    pub fn new(_left: f32, _bottom: f32, _width: f32, _height: f32) -> RectFloat {
        return RectFloat {
            left: _left,
            bottom: _bottom,
            width: _width,
            height: _height,
        };
    }

    // Tests whether _x and _y coordinates are within rectangle bounds
    pub fn is_inside(&self, _x: f32, _y: f32) -> bool {
        return _x >= self.left && _x < self.left + self.width && _y >= self.bottom &&
            _y < self.bottom + self.height;
    }

    // Gets the rectangle x coordinate of the right rectangle side.
    pub fn right(&self) -> f32 { return self.left + self.width; }

    // Gets the rectangle y coordinate of the top rectangle side.
    pub fn top(&self) -> f32 { return self.bottom + self.height; }
}

//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------
#[cfg(test)]
mod ozz_math {
    use crate::rect::*;

    #[test]
    fn rect_int() {
        let rect = RectInt::new(10, 20, 30, 40);

        assert_eq!(rect.right(), 40);
        assert_eq!(rect.top(), 60);

        assert_eq!(rect.is_inside(10, 20), true);
        assert_eq!(rect.is_inside(39, 59), true);

        assert_eq!(rect.is_inside(9, 20), false);
        assert_eq!(rect.is_inside(10, 19), false);
        assert_eq!(rect.is_inside(40, 59), false);
        assert_eq!(rect.is_inside(39, 60), false);
    }

    #[test]
    fn rect_float() {
        let rect = RectFloat::new(10.0, 20.0, 30.0, 40.0);

        assert_eq!(rect.right(), 40.0);
        assert_eq!(rect.top(), 60.0);

        assert_eq!(rect.is_inside(10.0, 20.0), true);
        assert_eq!(rect.is_inside(39.0, 59.0), true);

        assert_eq!(rect.is_inside(9.0, 20.0), false);
        assert_eq!(rect.is_inside(10.0, 19.0), false);
        assert_eq!(rect.is_inside(40.0, 59.0), false);
        assert_eq!(rect.is_inside(39.0, 60.0), false);
    }
}