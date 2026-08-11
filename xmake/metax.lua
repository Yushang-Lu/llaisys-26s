if not is_plat("linux") then
    raise("MetaX backend is only supported on Linux")
end

local maca_path = os.getenv("MACA_PATH") or "/opt/maca"
local cucc_path = os.getenv("CUCC_PATH") or path.join(maca_path, "tools", "cu-bridge")
local mxdriver_path = os.getenv("MXDRIVER_PATH") or "/opt/mxdriver"
local cucc = path.join(cucc_path, "bin", "cucc")

if not os.isfile(cucc) then
    raise("MetaX cucc compiler not found at " .. cucc)
end

METAX_LIBRARY_DIRS = {
    path.join(maca_path, "lib"),
    path.join(maca_path, "mxgpu_llvm", "lib"),
    path.join(mxdriver_path, "lib")
}

local function configure_metax_target()
    set_languages("cxx17")
    set_warnings("all", "error")
    set_policy("build.ccache", false)
    set_toolset("cxx", "clangxx@" .. cucc)
    add_includedirs(
        path.join(cucc_path, "include"),
        path.join(maca_path, "include", "mcblas"))
    add_cxflags(
        "-x", "maca",
        "-fPIC",
        "-Wno-unknown-pragmas",
        {force = true})
end

target("llaisys-device-metax")
    set_kind("static")
    configure_metax_target()

    add_files("../src/device/metax/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-ops-metax")
    set_kind("static")
    add_deps("llaisys-device-metax")
    add_deps("llaisys-tensor")
    configure_metax_target()

    add_files("../src/ops/*/metax/*.cpp")

    on_install(function (target) end)
target_end()
