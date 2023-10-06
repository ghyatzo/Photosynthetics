using Base.Threads
using Plots
using FFTW
using LinearAlgebra
using DSP
using BenchmarkTools
using StructArrays

gr()
theme(:orange)

sfft = fftshift ∘ fft
sifft = fftshift ∘ ifft
logplot = log10 ∘ abs2

const n = 1000 # 10 px per mm, 100 px per cm
const λ = 5e-3
const k = 2π/λ


# θx = 0
# θy = 0
# kx = k * sin(θx)
# ky = k * sin(θy)
# νx = kx / 2π
# νy = ky / 2π
# kz = sqrt(k^2 - kx^2 - ky^2) # forward propagation
# vk = [kx, ky, kz]

const half_length = 50 # half a size of the projector/ spatial domain
const spatial_range = range(-half_length, half_length, length=n+1) # millimiters
const mesh = [x for x in spatial_range, x in spatial_range]
const mesh_sum_sq = @. mesh^2 + $(mesh')^2

const sb = 1/λ # the biggest radius in frequence space without evanescent waves
const freq_range = range(-sb, sb, length=n+1)
const νmesh = [ν for ν in freq_range, ν in freq_range]
const νmesh_sum_sq = map((x) -> x < sb^2 ? x : 0, @. νmesh.^2 .+ νmesh'.^2)

const zmax = 5 # 2 meters away
const depth_range = range(-zmax, zmax, length=n+1)

struct Pupil
    z::Int64
    r::Float64
    ker::AbstractArray
    KER::AbstractArray
    Pupil(z, r) = begin
        ker = @. min(max(r^2 - mesh_sum_sq, 0), 1)
        new(z, r, ker, fft(ker))
    end
end

harmonic(F, νx, νy) = (x,y) -> F * exp(-im * 2π * (νx*x + νy*y))

function H(νρ, d) # transfer funciton of the free space
    @assert νρ <= 1/λ^2
    exp(im * 2π * sqrt(1/λ^2 - νρ) * d)
end

function h(ρ, d)
    (d/im*λ) * exp(im * k * sqrt(ρ + d^2)) / (ρ + d^2)
end

# Propagation matrices
Mhs = Array{Complex{Float64}}(undef, n+1, n+1, n+1)a
@inbounds @threads for k in axes(Mhs, 3)
    for j in axes(mesh_sum_sq, 2)
        for i in axes(mesh_sum_sq, 1)
            Mhs[i, j, k] = h(mesh_sum_sq[i,j], depth_range[k])
        end
    end
end

MHs = Array{Complex{Float64}}(undef, n+1, n+1, n+1)
@inbounds @threads for z in axes(Mhs, 3)
    MHs[:, :, z] = @views fft(Mhs[:, :, z])
end

function fr_prop(start, distance)
    entry = Int64(floor(distance/zmax * n))
    MH = fft(Mhs[:, :, entry])
    MF = fft(start)
    ifft(MF .* MH)
end

#gaussian
# Mf = exp.( -0.5 * mesh_sum_sq / 0.2)
Mf = [harmonic(1, 0.2, 0.1)(x,y) for y in spatial_range, x in spatial_range]
MF = fft(Mf)
heatmap(spatial_range, spatial_range, real.(Mf), grid=true)
heatmap(spatial_range, spatial_range, logplot.(fftshift(MF)), grid=true)

MGs = Array{Complex{Float64}}(undef, n+1, n+1, n+1)
@inbounds @threads for z in axes(MGs, 3)
    for j in axes(MGs, 2)
        for i in axes(MGs, 1)
            MGs[i, j, z] = @views MF[i, j] * MHs[i, j ,z]
        end
    end
end

Mgs = Array{Complex{Float64}}(undef, n+1, n+1, n+1)
sliceY = Array{Complex{Float64}}(undef, n+1, n+1)
sliceX = Array{Complex{Float64}}(undef, n+1, n+1)
@inbounds @threads for z in axes(MGs, 3)
    Mgs[:, :, z] = @views ifft(MGs[:, :, z])
    sliceY[:, z] = @views fftshift(Mgs[:, 50, z])
    sliceX[:, z] = @views fftshift(Mgs[50, :, z])
end

heatmap(spatial_range, spatial_range, abs2.(fftshift(Mgs[:, :, end])))
heatmap(depth_range, spatial_range, logplot.(sliceY))
