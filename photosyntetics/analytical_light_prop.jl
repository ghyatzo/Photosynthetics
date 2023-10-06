module AnalyticalLightProp

using Base.Threads
using Plots
using Gadfly
using FFTW
using LinearAlgebra
using DSP
using BenchmarkTools
using StructArrays

gr()
theme(:orange)

sfft  = fftshift ∘ fft
sifft = fftshift ∘ ifft
logplot = log10 ∘ abs2

const n = 100 # 10 px per mm, 100 px per cm
const λ = 0.0004
const k = 2π/λ

harmonic(F, νx, νy) = (x,y) -> F * exp(-im * 2π * (νx*x + νy*y))

# MADNESS
function U(x, y, ξ, η, σ, z, b, c, d)
    ss = 2*σ^2
    r2 = x^2 + y^2
    ρ2 = ξ^2 + η^2
    D = z+d
    Imk = im*k/2

    A = -b/ss + Imk*(c + inv(d))
    f1 = -ρ2/ss + Imk*r2/D
    f2 = (b*ξ/ss - Imk*x/D)
    f3 = (b*η/ss - Imk*y/D)

    den = im * λ * D

    return π * b/den * inv(A) * exp(f1 - inv(A) * (f2^2 + f3^2))
    # return (π * b)/im*λ*D * inv(A) * exp(f1 - inv(A) * (f2^2 + f3^2))
end

u(ξ, η, σ, b, c, d) = (x, y, z) -> U(x, y, ξ, η, σ, z, b, c, d)

half_length = 0.25 # half a size of the projector/ spatial domain
spatial_range = range(-half_length, half_length, length=n+1)
mesh = [x for x in spatial_range, x in spatial_range]
mesh_sum_sq = @. mesh^2 + $(mesh')^2

zmax = 14 # 2 meters away
depth_range = range(0, zmax, length=n+1)

b = -1
c = -1.01
d = -7.822

heatmap(gaussian((n, n), 0.1))
f = u(2, 1, 0.1, b, c, d)

# circle = [0.5 .* (cos(x), sin(y)) for x in range(0, 2π, length=n+1), y in range(0, 2π, length=n+1)]
const abstol = 1e5
travel = Array{ComplexF64}(undef, length(spatial_range), length(spatial_range), length(depth_range))
@inbounds @threads for z in axes(travel, 3)
    @inbounds for j in axes(travel, 2)
        @inbounds for i in axes(travel, 1)
            I = f(spatial_range[i], spatial_range[j], depth_range[z])
            I = abs(I) > abstol ? abstol : I
            I = abs(I) < -abstol ? 0 : I
            travel[i, j, z] = abs(I)
        end
    end
end

f(spatial_range[4], spatial_range[2], depth_range[3])

heatmap(logplot.(travel[:, n÷2, :]))

end
