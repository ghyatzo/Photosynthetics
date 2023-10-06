# module OneDimLightProp

using FFTW
using Base.Threads
using DSP
using Plots
gr()
theme(:orange)

n = 1000
λ = 0.0004
k = 2π/λ

half_length = 0.5
zmax = 14
xrange = range(-half_length, half_length, length=n+1)
zrange = range(0, zmax, length=n+1)

gauss(ξ, σ) = (x) -> exp(-inv(2σ^2)*(x-ξ)^2)
plot(xrange, gauss(0.14, 0.001).(xrange))

V(t, f::Function) = (x) -> sqrt(Complex(t)) * f( t * x)

Q(c, f::Function) = (x) -> exp(im * k/2 * c * x^2) * f(x)

function R(d, U::AbstractArray)
    ifft( Q(-λ^2*d, f).( fft( U )))
end

function get_params(a,b,f,g)
    r1 = g*b/(g-b)
    v1 = 1 - b/g
    q1 = (f - b + g)/(f * (b - g))

    d = r1 + inv(v1^2 * (1/a + q1))
    c = v1^2 * inv(inv(q1) + a) * (1 + a*q1)^2
    t = v1 * (1 + a*q1)

    (d, c, t)
end

d, c, t = get_params(1.01, 10, 1.01, 1.01)
f = gauss(0, 0.1)
plot(f.(xrange), xrange)
plot(xrange, abs2.(V(t, f).(xrange)))
plot(xrange, abs2.(Q(d, V(t, f)).(xrange)))


toprop = Q(c, V(t, f)).(xrange)
travel = Array{ComplexF64}(undef, n+1, n+1)
@inbounds @threads for z in axes(travel, 2)
    travel[:,z] = R(zrange[z] +d, toprop)
end

heatmap(logplot.(travel))

# end  # module OneDimLightProp
