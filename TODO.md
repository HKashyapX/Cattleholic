# TODO: Fix UnicodeDecodeError in train_model.py

## Steps to Complete:

1. **Update fix_encoding_issues.py**: Change the target directory from `data_test` to `data` to scan the correct folder. ✅ Completed
2. **Run fix_encoding_issues.py**: Execute the script to identify problematic files, create a backup, and remove them. ✅ Completed
3. **Test train_model.py**: Run the training script to verify that the error is resolved.
4. **Verify backup and restoration**: Confirm that a backup was created and can be used to restore files if needed.
5. **Monitor for issues**: Check if any other errors occur and address them if necessary.

## Notes:
- The script will create a backup of the `data` directory before removing problematic files.
- If you need to reverse the changes, you can restore from the backup directory (e.g., `data_backup`).
- After fixes, ensure all necessary image files are still present for training.
