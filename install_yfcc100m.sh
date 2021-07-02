mkdir yfcc100m
s3cmd get --recursive s3://mmcommons ./yfcc100m
tar -xvf yfcc100m/yfcc100m_exif.tgz -C ./yfcc100m
tar -xvf yfcc100m/yfcc100m_autotags-v1.tgz -C ./yfcc100m
tar -xvf yfcc100m/yfcc100m_dataset.tgz -C ./yfcc100m
tar -xvf yfcc100m/yfcc100m_hash.tgz -C ./yfcc100m
tar -xvf yfcc100m/yfcc100m_lines.tgz -C ./yfcc100m

rm -f yfcc100m/yfcc100m_exif.tgz
rm -f yfcc100m/yfcc100m_autotags-v1.tgz
rm -f yfcc100m/yfcc100m_dataset.tgz
rm -f yfcc100m/yfcc100m_hash.tgz
rm -f yfcc100m/yfcc100m_lines.tgz
