###################
# Ravi's Pipeline #
###################

# Imports
from Sections import *

# 1. Create FATI records for usable customers
generate_fati()

# 2. Create Customer Stage records for usable customers
generate_customer_stage()

# 3. Add month and year columns to CustomerStage
categorize_months_customer_stage()

# 4. Create a joined table with fati
categorize_months_fati()

# 5. Some CS code was written to generate the join ()

# 6. Normalize the categorical columns
add_columns_for_categories()

# 7. Normalize magnitudes
normalize_columns()

# 8. Normalize outputs
label_power_user_records()

# 9. Remove Un-necessary columns
remove_unnecessary_columns()

# 10. Show PCA plot
show_pca()

# 11. Do some training
run_random_forest()




print('Done All')