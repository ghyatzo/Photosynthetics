
using Plots
# using Gadfly
using Base.Threads
using BenchmarkTools
using FFTW

logplot = log10 ∘ abs2
sfft  = fftshift ∘ fft
sifft = fftshift ∘ ifft
# set_default_plot_size(15cm, 15cm)

const n = 100
const λ = 5e-4
const k = 2π/λ

half_length = 0.5
xrange = range(-half_length, +half_length, length=n+1)
yrange = range(-half_length, +half_length, length=n+1)
mesh = [(x, y) for y in yrange, x in xrange]
mesh_sum_sq = [p[1]^2+p[2]^2 for p in mesh]

function get_params(a,b,f,g)
    r1 = g*b/(g-b)
    v1 = 1 - b/g
    q1 = (f - b + g)/(f * (b - g))

    d = r1 + inv(v1^2 * (1/a + q1))
    c = v1^2 * inv(inv(q1) + a) * (1 + a*q1)^2
    t = v1 * (1 + a*q1)

    (d, c, t)
end

function gauss(p::Tuple, center::Tuple, σ )
    x, y = p
    ξ, η = center
    exp(-inv(2*σ^2) * ((x - ξ)^2 + (y - η)^2))
end

function V(b, f::Function)
    return (p) -> sqrt(Complex(b)) * f(b.*p)
end

function Q(c, U::AbstractArray)
    return @. U * exp(im * k/2 * c * mesh_sum_sq)
end

function Q!(c, U::AbstractArray)
    _exp = [exp(im * π/λ * c * p) for p in mesh_sum_sq]
    U .*= _exp
end

function R!(d, U::AbstractArray)
    ifft!(Q!(-λ^2*d, fft!(U)))
end

function R(d, U::AbstractArray)
    ifft(Q(-λ^2 * d, fft(U)))
end

f = 1.01 # 50 cm di focal distance
g = 1.01 # 1 m focal distance
b = 10 # 1 m of planar travel
c, d, t = get_params(1.01, b, f, g)


g0(center, σ) = (p) -> gauss(p, center, σ)
center = (0, 0)
σ = 0.1
heatmap(g0(center, σ).(mesh))

function U_test(z, psource, σ, d, c, t)
    g0 = (r) -> gauss(r, psource, σ)
    Vg0 = [V(t, g0)(p) for p in mesh]
    QVg0 = Q(c, Vg0)
    RQVg0 = R(z+d, QVg0)
end

function U_test2(ϵ, psource, σ, d, c, t)
    g0(center, σ) = (p) -> gauss(p, center, σ)
    return Q(c2, sfft( Q(c1, V(t, g0(psource, σ)).(mesh) )))
end

zmax = floor(abs(2*d))
zrange = range(0, zmax, length=n+1)
abstol = 1e3

travel = Array{ComplexF64}(undef, length(xrange), length(yrange), length(zrange))
@inbounds @threads for z in axes(travel, 3)
    travel[:, :, z] =  U_test(zrange[z], (0, 0), 0.1, d, c, t)
end

heatmap(zrange, xrange, logplot.(travel[:, 50, :]))

#
# zmax = -abs(d)
# zrange = range(0, 2*zmax, length=n+1)
# abstol = 1e3
#
# travel = Array{ComplexF64}(undef, length(xrange), length(yrange), length(zrange))
# @inbounds @threads for z in axes(travel, 3)
#     travel[:, :, z] =  U_test(zrange[z], (0, 0), 0.1, b, c, d)
# end
#
# heatmap(abs.(travel[:, 50, :]))
