if ! mysql -h db -uroot -e 'USE baseball'; then
  echo "Baseball database DOES NOT exist "
    mysql -h db -u root -e "CREATE DATABASE IF NOT EXISTS baseball"
  else
    echo "BASEBALL created"
