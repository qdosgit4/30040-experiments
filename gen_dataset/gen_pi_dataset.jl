using BFloat16s
using Distributions
using Random

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


function write_data(filename ::String, arr ::Vector{Tuple{BFloat16, Int64}})
    
    open(filename, "w") do io
        
        for t in arr

            println(io, Float64(t[1]), ",", t[2])
            
        end
        
    end

end


function main()

    q = 2000

    valid_pi, invalid_pi_below, invalid_pi_above = gen_pi_dataset(q)

    if minimum(invalid_pi_above) > maximum(valid_pi) && minimum(valid_pi) > maximum(invalid_pi_below)

        println("writing data")

        write_data("pi_dataset.txt", vcat(
            collect(zip(invalid_pi_above, zeros(Int, q))),
            collect(zip(valid_pi, ones(Int, q * 2))),
            collect(zip(invalid_pi_below, zeros(Int, q)))
        )
                   )

    end

end


main()

