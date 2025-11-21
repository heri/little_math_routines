
#[macro_use]
extern crate helix;

use helix::{CheckResult, CheckedValue, ToRuby, ToRust, UncheckedValue};
use helix::libc;
use helix::sys;
use std::collections::HashMap;
use std::error::Error;

#[derive(Debug, Clone)]
pub struct FloatVec(pub Vec<f64>);

extern "C" {
    fn rb_ary_new_capa(capacity: libc::c_long) -> sys::VALUE;
    fn rb_ary_push(array: sys::VALUE, item: sys::VALUE) -> sys::VALUE;
}

impl UncheckedValue<FloatVec> for sys::VALUE {
    fn to_checked(self) -> CheckResult<FloatVec> {
        unsafe {
            if !sys::RB_TYPE_P(self, sys::T_ARRAY) {
                return Err("Expected a Ruby Array".into());
            }
        }
        Ok(unsafe { CheckedValue::new(self) })
    }
}

impl ToRust<FloatVec> for CheckedValue<FloatVec> {
    fn to_rust(self) -> FloatVec {
        unsafe {
            let len = sys::RARRAY_LEN(self.inner) as isize;
            let ptr = sys::RARRAY_CONST_PTR(self.inner);

            let values = (0..len)
                .map(|idx| sys::NUM2F64(*ptr.offset(idx)))
                .collect();

            FloatVec(values)
        }
    }
}

impl ToRuby for FloatVec {
    fn to_ruby(self) -> sys::VALUE {
        unsafe {
            let array = rb_ary_new_capa(self.0.len() as libc::c_long);
            for value in self.0 {
                rb_ary_push(array, sys::F642NUM(value));
            }
            array
        }
    }
}

impl ToRuby for Result<f64, String> {
    fn to_ruby(self) -> sys::VALUE {
        match self {
            Ok(val) => val.to_ruby(),
            Err(msg) => helix::ExceptionInfo::with_message(msg).raise(),
        }
    }
}

pub fn h_distance(coord1: &[f64], coord2: &[f64]) -> f64 {
    if coord1.len() < 2 || coord2.len() < 2 {
        panic!("Both coordinates must have at least two elements.");
    }

    const EARTH_RADIUS_KM: f64 = 6371.0;

    let (lat1, lon1) = (coord1[0].to_radians(), coord1[1].to_radians());
    let (lat2, lon2) = (coord2[0].to_radians(), coord2[1].to_radians());

    let delta_lat = lat1 - lat2;
    let delta_lon = lon1 - lon2;

    let central_angle_inner = (delta_lat / 2.0).sin().powi(2)
        + lat1.cos() * lat2.cos() * (delta_lon / 2.0).sin().powi(2);

    let central_angle = 2.0 * central_angle_inner.sqrt().asin();

    EARTH_RADIUS_KM * central_angle
}

pub fn variance_f32(data: &[f64], mean: f64) -> f64 {
    let len = data.len();
    if len <= 1 {
        return 0.0; // Variance undefined
    }

    let numerator: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum();
    numerator / (len - 1) as f64
}

pub fn mean_f32(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    values.iter().sum::<f64>() / values.len() as f64
}

pub fn array_to_vec(arr: &[f64]) -> Vec<f64> {
     arr.to_vec()
}

pub fn median_f32(vect: &[f64]) -> f64 {
    if vect.is_empty() {
        panic!("Cannot compute median of an empty array.");
    }

    let mut numbers = vect.to_vec();
    numbers.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = numbers.len() / 2;

    if numbers.len() % 2 == 0 {
        (numbers[mid - 1] + numbers[mid]) / 2.0
    } else {
        numbers[mid]
    }
}

pub fn mode(vect: &[f64]) -> f64 {
    if vect.is_empty() {
        panic!("Cannot compute mode of an empty array.");
    }

    let mut occurrences: HashMap<u64, usize> = HashMap::new();

    for &value in vect {
        *occurrences.entry(value.to_bits()).or_insert(0) += 1;
    }

    occurrences
        .into_iter()
        .max_by_key(|&(_, count)| count)
        .map(|(bits, _)| f64::from_bits(bits))
        .expect("Unexpected error computing mode")
}

// Convenience alias to keep a stable Rust function name exposed to Ruby.
pub fn mode_f32(vect: &[f64]) -> f64 {
    mode(vect)
}


pub fn covariance_f32(x_values: &[f64], y_values: &[f64]) -> f64 {
    if x_values.len() != y_values.len() {
        panic!("x_values and y_values must have equal lengths.");
    }

    if x_values.is_empty() {
        return 0.0;
    }

    let mean_x = mean_f32(x_values);
    let mean_y = mean_f32(y_values);

    x_values
        .iter()
        .zip(y_values.iter())
        .map(|(&x, &y)| (x - mean_x) * (y - mean_y))
        .sum::<f64>()
        / x_values.len() as f64
}


pub struct LinearRegression {
    pub coefficient: Option<f64>,
    pub intercept: Option<f64>
}
 
impl LinearRegression {
    pub fn new() -> LinearRegression {
        LinearRegression { coefficient: None, intercept: None }
    }

    pub fn fit(&mut self, x_values: &[f64], y_values: &[f64]) -> Result<(), Box<dyn Error>> {
        if x_values.len() != y_values.len() || x_values.is_empty() {
            return Err("Input vectors must have the same non-zero length".into());
        }

        let x_mean = mean_f32(x_values);
        let y_mean = mean_f32(y_values);

        let covariance = covariance_f32(x_values, y_values);
        let variance = variance_f32(x_values, x_mean);

        if variance == 0.0 {
            return Err("Variance of x_values is zero, cannot perform regression".into());
        }

        let b1 = covariance / variance;
        let b0 = y_mean - b1 * x_mean;

        self.coefficient = Some(b1);
        self.intercept = Some(b0);
        Ok(())   
    }   

    pub fn predict(&self, x : f64) -> Result<f64, Box<dyn Error>> {
        match (self.coefficient, self.intercept) {
            (Some(b1), Some(b0)) => Ok(b0 + b1 * x),
            _ => Err("Model has not been fitted yet. Call `fit` first.".into()),
        }
    }

    pub fn predict_list(&self, x_values: &[f64]) -> Result<Vec<f64>, Box<dyn Error>> {
        x_values.iter().map(|&x| self.predict(x)).collect()
    }

    pub fn evaluate(&self, x_test: &[f64], y_test: &[f64]) -> Result<f64, Box<dyn Error>> {
        if x_test.len() != y_test.len() {
            return Err("Test vectors must have the same length".into());
        }

        let predictions = self.predict_list(x_test)?;
        root_mean_squared_error(y_test, &predictions)
    }
}

pub fn root_mean_squared_error(actual: &[f64], predicted: &[f64]) -> Result<f64, Box<dyn Error>> {
    if actual.len() != predicted.len() {
        return Err("Actual and predicted vectors must have the same length".into());
    }

    let mse = actual
        .iter()
        .zip(predicted.iter())
        .map(|(&a, &p)| (a - p).powi(2))
        .sum::<f64>()
        / actual.len() as f64;

    Ok(mse.sqrt())
}

pub fn standard_deviation_f32(data: &[f64], mean: f64) -> f64 {
    variance_f32(data, mean).sqrt()
}

pub fn max_f32(data: &[f64]) -> f64 {
    let mut iter = data.iter();
    let mut result = *iter
        .next()
        .expect("Cannot compute max of an empty array.");

    for &item in iter {
        if item > result {
            result = item;
        }
    }
    result
}

pub fn min_f32(data: &[f64]) -> f64 {
    let mut iter = data.iter();
    let mut result = *iter
        .next()
        .expect("Cannot compute min of an empty array.");

    for &item in iter {
        if item < result {
            result = item;
        }
    }
    result
}

pub fn max_usize(data: &[usize]) -> usize {
    let mut iter = data.iter();
    let mut result = *iter
        .next()
        .expect("Cannot compute max of an empty array.");

    for &item in iter {
        if item > result {
            result = item;
        }
    }
    result
}

pub fn min_usize(data: &[usize]) -> usize {
    let mut iter = data.iter();
    let mut result = *iter
        .next()
        .expect("Cannot compute min of an empty array.");

    for &item in iter {
        if item < result {
            result = item;
        }
    }
    result
}

ruby! {
    class LittleMath {
        def haversine_distance(coord1: FloatVec, coord2: FloatVec) -> f64 {
            return h_distance(&coord1.0, &coord2.0);
        }

        def mean(array: FloatVec) -> f64 {
            return mean_f32(&array.0);
        }

        // same as mean
        def average(array: FloatVec) -> f64 {
            return mean_f32(&array.0);
        }

        def variance(array: FloatVec, mean: f64) -> f64 {
            return variance_f32(&array.0, mean);
        }

        def covariance(array1: FloatVec, array2: FloatVec) -> f64 {
            return covariance_f32(&array1.0, &array2.0);
        }

        // currently this tries to fit x_Values and y_values with a simple linear regression and then uses model to predict for value
        def linear_reg(x_values: FloatVec, y_values: FloatVec) -> FloatVec {
            let mut model = LinearRegression::new();
            if let Err(e) = model.fit(&x_values.0, &y_values.0) {
                helix::ExceptionInfo::with_message(e.to_string()).raise();
            }

            FloatVec(vec![
                model.intercept.unwrap(),
                model.coefficient.unwrap()
            ])
        }

        def predict(x_values: FloatVec, intercept: f64, coefficient: f64) -> FloatVec {
            FloatVec(x_values.0
                .iter()
                .map(|&x| intercept + coefficient * x)
                .collect())
        }

        def evaluate(x_values: FloatVec, y_values: FloatVec, intercept: f64, coefficient: f64) -> f64 {
            let predicted: Vec<f64> = x_values.0
                .iter()
                .map(|&x| intercept + coefficient * x)
                .collect();

            match root_mean_squared_error(&y_values.0, &predicted) {
                Ok(val) => val,
                Err(e) => helix::ExceptionInfo::with_message(e.to_string()).raise(),
            }
        }

        def standard_deviation(array: FloatVec, mean: f64) -> f64 {
            return standard_deviation_f32(&array.0, mean);
        }

        def min(array: FloatVec) -> f64 {
            return min_f32(&array.0);
        }

        def max(array: FloatVec) -> f64 {
            return max_f32(&array.0);
        }

        def median(array: FloatVec) -> f64 {
            return median_f32(&array.0);
        }

        def mode(array: FloatVec) -> f64 {
            return mode_f32(&array.0);
        }
        
    }
}
