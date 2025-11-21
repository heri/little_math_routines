Little Math Routines in Rust, for your Ruby app

# About

https://studiozenkai.com/post/the-little-things-matter/

`little_math_routines` ensures performance improvements by offloading computation to Rust. It’s perfect for applications requiring fast, reliable calculations with large datasets. The methods are efficient for statistical operations, linear regression, geospatial distance computations, and more.

# Distance Calculations
* Haversine Distance - Distance between points on the Earth's surface.

```Ruby
require 'little_math'

math = LittleMath.new
coord1 = [40.7128, -74.0060] # New York City
coord2 = [34.0522, -118.2437] # Los Angeles

distance = math.haversine_distance(coord1, coord2)
puts "The distance between NYC and LA is #{distance.round(2)} km"
```

# Statistics
mean, variance
```Ruby
data = [10.0, 20.0, 30.0, 40.0]

mean = math.mean(data)
variance = math.variance(data, mean)

puts "Mean: #{mean}"
puts "Variance: #{variance}"
```

min, max, median, mode
```Ruby
data = [5.0, 2.0, 9.0, 2.0, 7.0]

min = math.min(data)
max = math.max(data)
median = math.median(data)
mode = math.mode(data)

puts "Min: #{min}, Max: #{max}, Median: #{median}, Mode: #{mode}"
```

Linear regression example
```Ruby
x_values = [1.0, 2.0, 3.0, 4.0]
y_values = [10.0, 20.0, 30.0, 40.0]

model = math.linear_reg(x_values, y_values)

# Extract coefficients (returns [intercept, coefficient])
intercept, coefficient = model

puts "Linear Regression Model: y = #{coefficient}x + #{intercept}"

# Predict values
predicted_y = coefficient * 5.0 + intercept
puts "Predicted value for x = 5.0: #{predicted_y}"
```

# Contributing

Bug reports and pull requests are welcome on GitHub at https://github.com/heri/little_math_routines .
