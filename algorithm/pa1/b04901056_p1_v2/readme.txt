1. 學號：
B04901056
2. 姓名：
張承洋
3. 使用之程式語言：< C++ >

4. 使用之編譯器：< GNU g++ >

5. 檔案壓縮方式: <zip -r b04901056_p1_v2.zip b04901056_p1_v2>

6. 各檔案說明： 
	src/parser.h 
	src/parser.cpp 
	src/XXXSort.cpp : 實作程式碼 
	report.doc：程式報告  
	XXXSort : Executable
binaries 
7. 編譯方式說明： 
	cd src
	g++ insertionSort.cpp parser.cpp -o ../insertionSort
	g++ mergeSort.cpp parser.cpp -o ../mergeSort
	g++ heapSort.cpp parser.cpp -o ../heapSort
	g++ quickSort.cpp parser.cpp -o ../quickSort
	       
8. 執行、使用方式說明：
  
	編譯完成後，在檔案目錄下會產生 xxxSort 的執行檔
   
	執行檔的命令格式為：
  ./xxxSort <input file name> <output file name>
   
	例如：要對 case1.dat 執行 merge sort 輸出 out.dat 
  
	則在命令提示下鍵入
 ./mergeSort case1.dat out.dat  


9. 執行結果說明（說明執行結果的觀看方法，及解釋各項數據等）：
詳見 report.doc  							
       

