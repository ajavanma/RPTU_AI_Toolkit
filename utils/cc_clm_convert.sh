for FILE in *.pcd; do flatpak run org.cloudcompare.CloudCompare -C_EXPORT_FMT ASC -SEP SEMICOLON -O $FILE -NO_TIMESTAMP -SAVE_CLOUDS; done
