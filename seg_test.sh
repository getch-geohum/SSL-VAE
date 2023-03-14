for fold in carpet grid wood leather tile
do
	for p in 95 96 97 98 99
	do
		for method in p-quantile max k-sigma
		do 
			python mvtec_test_accuracy.py --root D:/DATA/MVTec/AE --texture $fold --method $method --k 1 --p $p 
		done
	done
done

# python mvtec_test_accuracy.py --root D:/DATA/MVTec/AE --method p-quantile --k 1 --p 95
