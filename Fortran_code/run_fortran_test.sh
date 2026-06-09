#!/bin/bash

for i in {0..19}
do

  idx=$((i + 1))
  
  input_file="fortran_data_${idx}.dat"
  output_file="fortran_maps_output_${idx}.dat"

  if [[ -f "$input_file" ]]; then
    echo "=== Итерация i=$i (индекс файла: $idx) ==="
    
    cp "$input_file" fortran_data.dat
    echo "Копирован: $input_file -> fortran_data.dat"

    echo "Запуск ./prog..."
    ./prog

    if [[ -f "fortran_maps_output.dat" ]]; then
      cp fortran_maps_output.dat "$output_file"
      echo "Скопирован: fortran_maps_output.dat -> $output_file"
    else
      echo "Внимание: файл fortran_maps_output.dat не был создан после работы ./prog!"
    fi
    
  else
    echo "Ошибка: Входной файл $input_file не найден. Итерация пропущена."
  fi
  
  echo ""
done

echo "Все итерации завершены."