# using Debugger

using Plots

function gen_predictive_dist(d ::Int64, m ::Int64, n ::Int64, y ::Int64)

    alpha = zeros(UInt64, n)

    alpha_plus_d = zeros(UInt64, n)

    permutations_n = UInt128((2^m)^n)

    x_range = zeros(UInt64, permutations_n)

    res = zeros(Float32, permutations_n)

    for i = UInt128(0):UInt128(permutations_n - 1)

        alpha = gen_permutation(i, m, n)

        for j = 1:length(alpha)

            alpha_plus_d[j] = alpha[j] + d

        end

        # println(alpha)

        res[i+1] = (alpha[y] + d) / (sum(alpha_plus_d) + d*n)

        x_range[i+1] = i

        # @bp

        # println(i, " ", res[i+1])

    end

    return x_range, res

end


function gen_permutation(i ::UInt128, m ::Int64, n ::Int64)

    set = zeros(UInt64, n)

    and_bits = UInt128(2^m - 1)

    a = UInt128(0)

    for j = 0:(n-1)

        a = and_bits & i

        set[j+1] = a >> (j * m)
        
        and_bits = and_bits << m

    end

    return set
    
end


# println(gen_permutation(UInt128(0x1), 8, 8))

# println(gen_permutation(UInt128(0x1_0000_0000_0000), 8, 16))

#res = gen_predictive_dist(6, 5, 20, 1)

@time x_range, res = gen_predictive_dist(1, 3, 3, 1)

println(length(res))

# println(res)

p = bar(x_range, res, title="Dirichlet plot")

savefig(p, "bar.pdf")
