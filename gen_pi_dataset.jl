using BFloat16s
using Distributions

function gen_pi_dataset(n ::Int64)

    valid_pi = zeros(BFloat16, n * 2)

    invalid_pi_below = zeros(BFloat16, n)

    invalid_pi_above = zeros(BFloat16, n)

    dataset = zeros(BFloat16, n * 2)

    pi_bf = BFloat16(pi)

    range = 0.025

    gaussian = Normal(0.0, range)

    gaussian_2 = Normal(0.0, 1)

    map!(x -> pi_bf + rand(gaussian), valid_pi, valid_pi)

    map!(x -> pi_bf + (range*5 + abs(rand(gaussian_2))), invalid_pi_above, invalid_pi_above)

    map!(x -> pi_bf + (-1 * (range*5 + abs(rand(gaussian_2))) ), invalid_pi_below, invalid_pi_below )

    # println(valid_pi)

    valid_pi, invalid_pi_below, invalid_pi_above
    
end


valid_pi, invalid_pi_below, invalid_pi_above = gen_pi_dataset(10000)

         
         

println(maximum(invalid_pi_above))
println(minimum(invalid_pi_above))

println(maximum(valid_pi))
println(minimum(valid_pi))


println(maximum(invalid_pi_below))
println(minimum(invalid_pi_below))


