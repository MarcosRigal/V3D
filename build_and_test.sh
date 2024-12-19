#!/bin/bash

# List of directories
folders=("P1" "P2" "P3" "P4" "P5")

# Check if a specific folder is passed as an argument
if [ "$1" ]; then
    if [[ " ${folders[@]} " =~ " $1 " ]]; then
        folders=($1) # Set folders to only the specified folder
    else
        echo "Error: Folder $1 is not in the predefined list."
        exit 1
    fi
fi

# Initialize result files
build_results_file="build_results.log"
test_results_file="test_results.log"
echo "Build Results Summary:" > "$build_results_file"
echo "======================" >> "$build_results_file"
echo "Test Results Summary:" > "$test_results_file"
echo "=====================" >> "$test_results_file"

# Loop through each folder
for folder in "${folders[@]}"; do
    echo "Processing folder: $folder"

    # Navigate to the folder
    if cd "$folder"; then
        # Create and navigate to the build directory
        mkdir -p build && cd build

        # Run cmake and make, capturing build errors if any
        echo "Building $folder..."
        cmake .. >> "../../$build_results_file" 2>&1 && make >> "../../$build_results_file" 2>&1
        if [ $? -eq 0 ]; then
            echo "[$folder] Build successful." >> "../../$build_results_file"
        else
            echo "[$folder] Build failed." >> "../../$build_results_file"
            cd ../..
            continue # Skip to the next folder if build fails
        fi

        echo "======================" >> "../../$build_results_file"

        # Run the test and capture its output
        echo "Running test for $folder..."
        ./test_common_code >> "../../$test_results_file" 2>&1
        if [ $? -eq 0 ]; then
            echo "[$folder] Test passed." >> "../../$test_results_file"
        else
            echo "[$folder] Test failed." >> "../../$test_results_file"
        fi
        echo "======================" >> "../../$test_results_file"

        # Return to the parent folder and clean up
        cd ..
        rm -rf build

        # Return to the base directory
        cd ..
    else
        echo "[$folder] Failed to navigate to $folder. Skipping." >> "$build_results_file"
    fi

done

# Clean up individual logs
for folder in "${folders[@]}"; do
    rm -f "${folder}_build.log" "${folder}_test.log"
done
