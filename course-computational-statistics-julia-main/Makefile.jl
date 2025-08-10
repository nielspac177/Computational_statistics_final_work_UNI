# Load packages

using ComposerTools

# Update Manifest.toml and Project.toml

copyproject()

# Create scripts by merging input scripts inside each folder

createscripts(joinpath("..", "..", "scripts", "code"), "scripts", remove = true)

# Compile scripts to notebooks

createnotebooks("scripts", "notebooks")

# # Create markdown files
# repo_path = "https://github.com/ErickChacon/01-computational-statistics-julia/blob/main"
# rm(joinpath("docs", "src"), recursive = true, force = true)
# Literate.markdown.(jls, joinpath("docs", "src"), execute = true, documenter = true,
#     repo_root_url = repo_path, credit = false)
