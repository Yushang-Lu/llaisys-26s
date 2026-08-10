target("llaisys-device-nvidia")
    set_kind("static")
    set_languages("cxx17")
    set_warnings("all", "error")
    set_values("cuda.rdc", false)

    add_files("../src/device/nvidia/*.cu")
    add_cugencodes("native", "compute_80")
    if not is_plat("windows") then
        add_cuflags("-Xcompiler=-fPIC")
    end

    add_links("cudart", {public = true})

    on_install(function (target) end)
target_end()

target("llaisys-ops-nvidia")
    set_kind("static")
    add_deps("llaisys-device-nvidia")
    add_deps("llaisys-tensor")

    set_languages("cxx17")
    set_warnings("all", "error")
    set_values("cuda.rdc", false)

    add_files("../src/ops/*/nvidia/*.cu")
    add_cugencodes("native", "compute_80")
    if not is_plat("windows") then
        add_cuflags("-Xcompiler=-fPIC")
    end

    add_links("cublas", "cudart", {public = true})

    on_install(function (target) end)
target_end()
