using BFloat16s
using Distributions
using Random
using Plots
using StatsBase

function gen_pi_dataset(n ::Int64)

    valid_pi = zeros(BFloat16, n * 2)

    invalid_pi_below = zeros(BFloat16, n)

    invalid_pi_above = zeros(BFloat16, n)

    dataset = zeros(BFloat16, n * 2)

    pi_bf = BFloat16(pi)

    range = 0.025

    gaussian = Normal(0.0, range)

    gaussian_2 = Normal(0.0, 1)

    # map!(x -> pi_bf + rand(gaussian), valid_pi, valid_pi)

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


function plot_data(arr ::Array{BFloat16})

    dataset = map!(x -> Float64(x), arr)

    # print(dataset)

    bins = minimum(dataset):0.05:maximum(dataset)

    print(bins)

    hist = fit(Histogram, dataset, bins)

    bar_chart = plot(hist, legend=false, xlabel="Value", ylabel="Frequency", title="Bar Chart of Intervals and Frequencies")

    savefig(bar_chart, "bar_chart.png")

    savefig(bar_chart, "bar_chart.pdf")

    savefig(bar_chart, "bar_chart.svg")
    
end


function main()

    q = 500

    valid_pi, invalid_pi_below, invalid_pi_above = gen_pi_dataset(q)

    if minimum(invalid_pi_above) > maximum(valid_pi) && minimum(valid_pi) > maximum(invalid_pi_below)

        println("plotting data")

        plot_data(vcat(invalid_pi_above, valid_pi, invalid_pi_below))

        # println("writing data")

        # write_data("pi_dataset.txt", vcat(
        #     collect(zip(invalid_pi_above, zeros(Int, q))),
        #     collect(zip(valid_pi, ones(Int, q * 2))),
        #     collect(zip(invalid_pi_below, zeros(Int, q)))
        # )
        #            )

    end

end


main()

