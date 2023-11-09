select * from department;
select * from employee;
select * from project;
select * from works_on;

exec GetAllEmployees

-- employe with join project they work on & employee with no project

select e.emp_fname, e.emp_lname, j.job from employee e left Join  works_on j 
on e.emp_no=j.emp_no 

-- employee name, proj name, budget and budget >140000

select e.emp_fname, e.emp_lname, p.budget, p.project_name 
from employee e, project p, works_on w
where e.emp_no =w.emp_no
and p.project_no= w.project_no
and p.budget >140000;

select e.emp_fname, e.emp_lname,w.job from employee e 
join works_on w on e.emp_no=w.emp_no
join project p on p.project_no=w.project_no
where p.budget > 140000

-- employe , project name and budget, show all employee who has and has not project

select e.emp_fname, e.emp_lname,w.job, p.budget from employee e 
left join works_on w on e.emp_no=w.emp_no
left join project p on p.project_no=w.project_no

-- subquery-- always in (*query*)
-- noncorrelated -> subquery which can run on its own and it will execute first
-- correlated
-- scalar- single value return back
-- multirow subquery
-- derived table -> goes in from clause

-- non correlated -- 

-- return employe name  and at least one project name

select * from employee 
where emp_no IN
(select emp_no from works_on);

-- 
insert into works_on values(15000, 'p2','SQL Developer','2021-11-10')

--employe name with atleast 2 jobs 

select * from employee e where e.emp_no IN
(select emp_no from works_on 
group by emp_no
having count(*)>=2
)

-- emp name with no project

select emp_fname, emp_lname from employee where emp_no not IN
(select emp_no from works_on)

select * from employee e where NOT EXISTS (select * from works_on w where w.emp_no =e.emp_no)

-- conditional Expression--
-- here value will be added in state 
-- new column is created 
-- after then value "data" will added in new column
/*
select *

case 1
when '' then ''
end as [new columnName], 

case 2
when '' then ''
end as [new columnName]

from table_name
*/

select *,
CASE [location] 
WHEN 'Dallas' THEN 'TX'
WHEN 'Seattle' THEN 'WA'
WHEN 'Boston' THEN 'MA'
WHEN 'New York' THEN 'NY'
when 'London' THEN 'n/a'
END AS [State],
CASE [location] 
WHEN 'Dallas' THEN 'USA'
WHEN 'Seattle' THEN 'USA'
WHEN 'Boston' THEN 'USA'
WHEN 'New York' THEN 'USA'
when 'London' THEN 'United Kingdom'
end as [Country]
from department

select *,
CASE [location] 
WHEN 'Dallas' THEN 'TX'
WHEN 'Seattle' THEN 'WA'
WHEN 'Boston' THEN 'MA'
WHEN 'New York' THEN 'NY'
when 'London' THEN 'n/a'
END AS [State],
CASE [location] 
when 'London' THEN 'United Kingdom'
else 'USA'
end as [country]
from department

-- Union--
-- column order should be same
-- union all allow duplicate
-- union no duplicate

--union employee  with no project with emp more the 1 project

select emp_fname, emp_lname, 'No Work' as Status from employee where emp_no not IN
(select emp_no from works_on)
union
select emp_fname, emp_lname, 'More than 1 Job' as Status from employee e where e.emp_no IN
(select emp_no from works_on 
group by emp_no
having count(*)>1)

-- SQL/PSM  procedural language
-- case, if, loop, for, while etc
--routines and triggers
-- routins -> UDF 
-- function -> return value on i/p para
-- procedure -> no return justi/p and o/p'
-- store prcedure can call directly but not trigger

-- dynamic SQL alternative for Stored procedure
-- Embedded SQL
-- views and UDf
-- ORM tools -> object oriented to SQL
    -- translate object oriented code to SQ


CREATE PROCEDURE GetAllEmployees As
BEGIN
/* all logic goes here*/
print 'Hello World'

END

-- show employee table

Alter PROCEDURE GetAllEmployees As
BEGIN

 Select * from employee

END

-- departmentwise employee

Alter PROCEDURE GetAllEmployees @dept_no varchar(5) As
BEGIN

 Select e.emp_fname, e.emp_lname, e.dept_no 
 from employee e
 where e.dept_no=@dept_no

END

exec GetAllEmployees 'D1'

-- emp with larget emp id

Alter PROCEDURE GetAllEmployees As
BEGIN

 Select TOP(1) e.emp_no 
 from employee e
 order by emp_no asc
 
END

-- new

CREATE PROCEDURE GetEmployee AS
DECLARE @last_employee int

BEGIN
	SELECT @last_employee = max(emp_no) FROM employee
	SELECT e.emp_fname, e.emp_lname, e.dept_no
	FROM employee e
	WHERE e.emp_no = @last_employee
END

exec GetEmployee
