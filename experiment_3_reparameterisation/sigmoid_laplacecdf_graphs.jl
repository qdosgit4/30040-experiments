using Plots
using Distributions

##  Generated via LLM (but makes heavy use of Julia libraries regardless).

# 1. Define the functions
# Parameterised Sigmoid
sigmoid_param(x, k) = 1 / (1 + exp(-k * x))

# Laplace CDF with mu=0, b=1
laplace_dist = Laplace(0.0, 0.018)

# 2. Create the x range
x = -0.2:0.005:0.2

# 3. Calculate values
y_sigmoid = sigmoid_param.(x, 75)          # Broadcast over the array
y_laplace = cdf.(laplace_dist, x) # Broadcast the CDF function

# 4. Generate the Plot
p = plot(x, y_sigmoid, 
     label="Parameterised Sigmoid (75)", 
     linewidth=2, 
     legend=:bottomright)

plot!(x, y_laplace, 
      label="Laplace CDF (0, 0.018)", 
      linewidth=2)

xlabel!("x")
ylabel!("Probability")

savefig("sigmoid_laplace_cdf.pdf")
