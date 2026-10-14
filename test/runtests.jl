using Test, StoPEPS

# Smoke check for package loading and default configuration.
@testset "StoPEPS smoke check" begin
    config = RunConfig()
    @test validate_config(config) === config
end
