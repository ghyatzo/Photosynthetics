using Base.Threads
using Plots
using FFTW
using LinearAlgebra
using DSP
using BenchmarkTools
using StructArrays

gr()
theme(:juno)

const n = 100
 Ω = [0, 0.5]
const λ = 5e-3
const k = 2π/λ

const xs = range(0, 0.5, length=n+1)
const zmax = 10
sfft = fftshift ∘ fft

a0 = [sin(2π*20*x + π/3) for x in xs]
l = length(a0)
plot(xs, a0)
A0 = fft(a0)
plot(abs.(A0)) # raw absolute value FFT
plot(range(0, 1, length=l), abs2.(A0)) # raw absolute value FFT with normalised "frequences"
plot(range(-l/2, l/2, length=l), abs2.(fftshift(A0))) #shifter FFT with normalised frequences, so that the 0th element of the vector is in the center
plot(range(-l/2, l/2, length=l), abs2.(fftshift(A0))./l)
plot(range(0, l/2-1, length=l÷2), abs2.(A0[1:n÷2])./l) # Direction of the wave



θx = 0
θy = 0
kx = k * sin(θx)
ky = k * sin(θy)
νx = kx / 2π
νy = ky / 2π
kz = sqrt(k^2 - kx^2 - ky^2) # forward propagation
vk = [kx, ky, kz]
