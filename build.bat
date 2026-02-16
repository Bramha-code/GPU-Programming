@echo off
REM Build script for Windows

echo ===================================
echo GPU Programming - Build Script
echo ===================================
echo.

REM Check if NVCC is available
where nvcc >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: NVCC not found in PATH
    echo Please install CUDA Toolkit and add to PATH
    echo Download from: https://developer.nvidia.com/cuda-downloads
    pause
    exit /b 1
)

echo CUDA Toolkit found:
nvcc --version
echo.

REM Create output directory
if not exist "bin" mkdir bin

REM Compilation flags
set NVCC_FLAGS=-O3 -arch=sm_75

echo Building examples...
echo.

REM Build basic examples
echo [1/4] Building vector addition...
nvcc %NVCC_FLAGS% 01_basics\add.cu -o bin\add.exe
if %ERRORLEVEL% NEQ 0 goto :error

echo [2/4] Building vector addition benchmark...
nvcc %NVCC_FLAGS% 01_basics\add_bench.cu -o bin\add_bench.exe
if %ERRORLEVEL% NEQ 0 goto :error

echo [3/4] Building matrix-vector multiplication...
nvcc %NVCC_FLAGS% 01_basics\matvec.cu -o bin\matvec.exe
if %ERRORLEVEL% NEQ 0 goto :error

echo [4/4] Building matrix multiplication...
nvcc %NVCC_FLAGS% 01_basics\matmul.cu -o bin\matmul.exe
if %ERRORLEVEL% NEQ 0 goto :error

echo.
echo ===================================
echo Build completed successfully!
echo ===================================
echo.
echo Executables are in the 'bin' directory:
dir /B bin\*.exe
echo.
echo To run an example:
echo   bin\add.exe
echo   bin\matmul.exe
echo   bin\add_bench.exe
echo.
pause
exit /b 0

:error
echo.
echo ===================================
echo Build failed!
echo ===================================
pause
exit /b 1
