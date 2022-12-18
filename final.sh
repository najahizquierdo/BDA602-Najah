#!/usr/bin/env bash
python3 -m pip install --upgrade pip
sleep 10

echo "Creating baseball database..."
mysql -h db -u root -ppassword -e "CREATE DATABASE IF NOT EXISTS baseball"
echo "loading baseball SQL file..."
mysql -h db -u root -ppassword baseball < ./baseball.sql
echo "using baseball..."
mysql -h db -u root -ppassword -e "USE baseball;"
echo "Running SQL scripts..."
mysql -h db -u root -ppassword baseball < ./finalbaseball.sql

python3 ./main.py