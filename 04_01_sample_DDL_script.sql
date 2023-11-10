-- iF DATABASE EXISTS, DROP
IF EXISTS (SELECT name FROM sys.databases WHERE name = N'sample')
    DROP DATABASE sample
GO

CREATE DATABASE [sample]
go
USE [sample]
GO

CREATE TABLE [dbo].[department](
	[dept_no] [char](4) NOT NULL,
	[dept_name] [varchar](25) NOT NULL,
	[location] [varchar](25) NULL,
 CONSTRAINT [prim_dept] PRIMARY KEY CLUSTERED 
([dept_no] ASC
)
) ON [PRIMARY]
GO

CREATE TABLE [dbo].[employee](
	[emp_no] [int] NOT NULL,
	[emp_fname] [varchar](20) NOT NULL,
	[emp_lname] [varchar](20) NOT NULL,
	[dept_no] [char](4) NULL,
 CONSTRAINT [prim_emp] PRIMARY KEY CLUSTERED 
(
	[emp_no] ASC
)
) ON [PRIMARY]
GO

ALTER TABLE [dbo].[employee]  WITH CHECK ADD  CONSTRAINT [foreign_emp] FOREIGN KEY([dept_no])
REFERENCES [dbo].[department] ([dept_no])
GO
ALTER TABLE [dbo].[employee] CHECK CONSTRAINT [foreign_emp]
GO


CREATE TABLE [dbo].[project](
	[project_no] [char](4) NOT NULL,
	[project_name] [varchar](50) NULL,
	[budget] [float] NULL,
 CONSTRAINT [prim_proj] PRIMARY KEY CLUSTERED 
(
	[project_no] ASC
)
) ON [PRIMARY]
GO

CREATE TABLE [dbo].[works_on](
	[emp_no] [int] NOT NULL,
	[project_no] [char](4) NOT NULL,
	[job] [varchar](50) NULL,
	[enter_date] [date] NULL,
 CONSTRAINT [prim_works] PRIMARY KEY CLUSTERED 
(
	[emp_no] ASC,
	[project_no] ASC
)
) ON [PRIMARY]
GO

ALTER TABLE [dbo].[works_on]  WITH CHECK ADD  CONSTRAINT [foreign1_works] FOREIGN KEY([emp_no])
REFERENCES [dbo].[employee] ([emp_no])
GO

ALTER TABLE [dbo].[works_on] CHECK CONSTRAINT [foreign1_works]
GO

ALTER TABLE [dbo].[works_on]  WITH CHECK ADD  CONSTRAINT [foreign2_works] FOREIGN KEY([project_no])
REFERENCES [dbo].[project] ([project_no])
GO

ALTER TABLE [dbo].[works_on] CHECK CONSTRAINT [foreign2_works]
GO



