#标注样本的路径
labmeDir=/trainData/XieTou/CZJT-20221118
outDir=/trainData
#标注样本当前最大类别数（labmel标注中最大的label名+1）
nc=10
zq_times=2
dataType=XTYL
name_xc=qzjt

#训练后的模型存放路径
outModelDir=${outDir}/${dataType}_model
#现场名字

#是否统计一下样本分布
#ifCall=true
#临时路径
tmpDir=${outDir}/${dataType}/tmp_data

#if [${ifCall}]; then
#python datapre/labelme2yoloByParameter_tj.py \
#--json_path $labmeDir \
#--dataset_save_path tmp_yolo \
#--label_overlook '-1'
#exit
#fi


#dist/labelme2yoloByParameter_tj/labelme2yoloByParameter_tj \
#--json_path $labmeDir \
#--dataset_save_path tmp_yolo \
#--label_overlook '-1'
#exit

#for i in {1..30} ; do
#    ps aux|grep train|awk '{print $2}'|xargs kill -9
#done

rm -rf $tmpDir
mkdir $tmpDir

dist/preData/preData  \
--labelme_path $labmeDir \
--save_path $tmpDir \
--min_kc 50  \
--max_kc 500  \
--zq_times ${zq_times} \
--qz_str ${dataType}



rm -rf $tmpDir/yolo_data
dist/labelme2yoloByParameter/labelme2yoloByParameter \
--json_path $tmpDir \
--dataset_save_path $tmpDir/yolo_data \
--label_overlook '-1' \
--type 'train'

dist/labelme2yoloByParameter/labelme2yoloByParameter \
--json_path $labmeDir \
--dataset_save_path $tmpDir/yolo_data \
--label_overlook '-1' \
--type 'train'

dist/labelme2yoloByParameter/labelme2yoloByParameter  \
--json_path $labmeDir \
--dataset_save_path $tmpDir/yolo_data \
--label_overlook '-1' \
--type 'val'






#训练
for i in {1..3} ; do
  rm -rf $tmpDir/run/train/
dist/auto_train/auto_train  \
--nc $nc \
--trainDir $tmpDir/yolo_data/images/train  \
--valDir $tmpDir/yolo_data/images/val  \
--epochs 2000 \
--batch-size 128  \
--imgsz 1024 \
--device '0,1,2,3' \
--workers 4 \
--project  $tmpDir/run/train/
done



rm -rf $outModelDir
if [ ! -d "$outModelDir" ]; then
    mkdir "$outModelDir"
    echo "build "${outModelDir}
fi
ls_date=`date +%Y%m%d%H%M`
mkdir -p $outModelDir/$ls_date
cp $tmpDir/run/train/exp/weights/best.pt $outModelDir/$ls_date
cp $tmpDir/run/train/exp/results.csv $outModelDir/$ls_date
sleep 1s
#统计map
python calResult.py \
--resultCSVPath $outModelDir/$ls_date/results.csv \
--mapPath $outModelDir/$ls_date/map.txt
sleep 1s
map=$(awk -F " " '{print $2}' $outModelDir/$ls_date/map.txt)
outname=${dataType}_${name_xc}_${ls_date}_${map}_3_1024_1024_${nc}.pt
echo $outname
cp $outModelDir/$ls_date/best.pt $outModelDir/$ls_date/$outname

#转化模型
python export.py --weights $outModelDir/$ls_date/$outname


