CREATE DATABASE cleancode;

CREATE USER 'participant'@'%' IDENTIFIED BY '<see keepass>';
GRANT SELECT ON cleancode.* TO 'participant'@'%' WITH GRANT OPTION;