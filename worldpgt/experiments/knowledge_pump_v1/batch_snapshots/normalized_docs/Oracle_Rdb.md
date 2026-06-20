# Oracle Rdb

Source: https://en.wikipedia.org/wiki/Oracle_Rdb
Retrieved at: 2026-06-20T08:05:43Z
Revision ID: 1329264079
Raw text SHA256: 8e2f93896a2e646154cf984b5ece4795c6a9c32fd310de7b400844b0b9c49405
Status: LOCAL_WIKIPEDIA_SNAPSHOT
Safe for accepted memory: false
Requires ingestion/quarantine/promotion/regression: true
Oracle Rdb is a relational database management system for the OpenVMS operating system.  It was originally released by Digital Equipment Corporation (DEC) in 1984 as VAX Rdb/VMS.

Product history
Rdb was a component of the VAX Information Architecture, and was designed to interoperate with other Digital database tools and application frameworks such as the Application Control Management System, Datatrieve and the Common Data Dictionary. It originally provided a proprietary query interface known as the Relational Data Operator (RDO), but later gained support for ANSI SQL.
In 1994 DEC sold the Rdb division to Oracle Corporation where it was rebranded Oracle Rdb. As of 2020, Oracle is still actively developing Rdb, with over half of the codebase developed under Oracle's ownership. It remains a VMS-only product; version 7.0 runs on OpenVMS for VAX and Alpha, version 7.1 on Alpha only, and versions 7.2 to 7.4 on Alpha and IA-64 (Itanium).
Rdb featured one of the first cost-based optimizers, and after acquisition Oracle introduced a cost-based optimizer in its regular Oracle RDBMS product.
On March 22, 2011, Oracle announced it had decided to end all software development on the Itanium, and that Oracle Rdb 7.3 would be the last major version released by Oracle. Due to a lawsuit filed by HP against Oracle, Oracle was ordered to continue porting its software to Itanium computers for as long as HP (now Hewlett Packard Enterprise) sells Itanium computers.
Despite the announcement that 7.3 would be the last major release, Oracle released version 7.4.1.0 of Rdb in August 2020 for OpenVMS on both Alpha and Itanium. In November 2020, Oracle announced that they are in the process of porting Rdb 7.4 to the x86-64 port of OpenVMS.

Data access
Interactive access to the Oracle Rdb can be by SQL (Structured Query Language), RDO (Relational Database Operator), or both.
High level languages usually access Oracle-Rdb by:

embedding RDO statements in the source file then running it through a precompiler
(example: "file.RCO" is pre-compiled into "file.COB")
embedding SQL statements in the source file then running it through a precompiler
(example: "file.SCO" is pre-compiled into "file.COB")
placing the SQL statements in a file external to the source code; this separate file is converted to object code by the "SQL Module Language" compiler, and the source code then references these SQL statements and, after compilation, the two are joined by the OpenVMS linker.
example: $ SQL$MOD file_bas.sqlmod       → file_bas.obj
$ BASIC   file.bas              → file.obj
$ LINK    file.obj,file_bas.obj → file.exe

A variation of example 3 allows "Dynamic SQL" to be created in the source code, and then used to communicate with Rdb via a structure known as SQLDA (SQL Descriptor Area).
On OpenVMS systems, Oracle Rdb is a popular (although expensive) upgrade path for applications written using Record Management Services (RMS) files.

Architecture
Rdb is built on top of a low-level database kernel named KODA, which handles functionality such as locking, journaling, and buffering of data. The KODA kernel is shared with Oracle's CODASYL DBMS (originally known as VAX DBMS) which is a network model database.

Rdb on other platforms
VAX Rdb/ELN was the name of Digital's relational database for the VAXELN operating system. Despite sharing the Rdb name, and being announced at the same time, Rdb/ELN was not based on Rdb/VMS, or vice versa. Rdb/ELN was created by Jim Starkey, and was the first commercially available database to use Multiversion concurrency control.
Ports of Rdb previously existed or were planned for Tru64 and Microsoft Windows NT.  Demand for the Tru64 version was so low that support was dropped. The Windows NT port was never released as Oracle could not obtain support on the BLISS compiler necessary for this platform. In order to port Rdb to these platforms, an abstraction layer named the Common Operating System Interface (COSI) was implemented to isolate the database from the underlying operating system.
Digital provided a relational database for their Ultrix operating system named ULTRIX/SQL, but it was based on Ingres instead of Rdb.

References

External links
Oracle-Rdb home page
OpenVMS Notes: Oracle Rdb
Free OpenVMS BASIC Rdb educational software
Oracle Rdb listserver discussion
