# Adding a print statement to indicate the start of the script
print("Starting script...")

# Checking if the 'mpmi' package is installed
if (!requireNamespace("mpmi", quietly = TRUE)) {
  print("Installing mpmi package...")
  install.packages("mpmi", repos = "https://cloud.r-project.org/")
}

# Adding a print statement to indicate the package has been loaded
print("Loading mpmi package...")
library(mpmi)

# Load your data
print("Loading data...")
word_data <- read.csv("/om/user/luwo/projects/MIT_prosody/notebooks/words_prominence.csv")

# To access 'word' and 'prominence' columns
disc_data <- word_data$word
cts_data <- word_data$prominence

# Set how much data to pass to mmi function
data_length <- 100000  # you can change this value
disc_data <- disc_data[1:data_length]
cts_data <- cts_data[1:data_length]

# Convert data to matrix form if needed
print("Converting data to matrix...")
cts_matrix <- as.matrix(cts_data)
disc_matrix <- as.matrix(disc_data)

# Run the mmi function
print("Running MMI...")
result <- mmi(cts = cts_matrix, disc = disc_matrix, level = 3, na.rm = FALSE)

# Output the result
print("MMI result:")
print(result)
