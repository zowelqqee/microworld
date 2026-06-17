# Web service

Source: https://en.wikipedia.org/wiki/Web_service
Retrieved at: 2026-06-17T17:17:50Z
Revision ID: 1345273563
Raw text SHA256: 24572bb7bfcfc5fff20b0fbafd382420a345e5b5bc63f4f8d5688d026b86ea21
Status: LOCAL_WIKIPEDIA_SNAPSHOT
Safe for accepted memory: false
Requires ingestion/quarantine/promotion/regression: true
A web service (WS) is either:

a service offered by an electronic device to another electronic device, communicating with each other via the Internet, or
a server running on a computer device, listening for requests at a particular port over a network, serving web documents (HTML, JSON, XML, images).
In a web service, a web technology such as HTTP is used for transferring machine-readable file formats such as XML and JSON.
In practice, a web service commonly provides an object-oriented web-based interface to a database server, utilized for example by another web server, or by a mobile app, that provides a user interface to the end-user. Many organizations that provide data in formatted HTML pages will also provide that data on their server as XML or JSON, often through a Web service to allow syndication. Another application offered to the end-user may be a mashup, where a Web server consumes several Web services at different machines and compiles the content into one user interface.

Web services (generic)

Asynchronous JavaScript and XML

Asynchronous JavaScript and XML (AJAX) is a dominant technology for Web services. Developing from the combination of HTTP servers, JavaScript clients and Plain Old XML (as distinct from SOAP and W3C Web Services), now it is frequently used with JSON as well as, or instead of, XML.

REST

Representational State Transfer (REST) is an architecture for well-behaved Web services that can function at Internet scale.
In a 2004 document, the W3C sets following REST as a key distinguishing feature of Web services:

We can identify two major classes of Web services:
REST-compliant Web services, in which the primary purpose of the service is to manipulate XML representations of Web resources using a uniform set of stateless operations; and
arbitrary Web services, in which the service may expose an arbitrary set of operations.

Technologies and protocols related to web services using markup languages
There are a number of Web services that use markup languages:

JSON-RPC.
JSON-WSP
Representational state transfer (REST) versus remote procedure call (RPC)
Simple Object Access Protocol (SOAP)
Web Services Conversation Language (WSCL)
Web Services Description Language (WSDL), developed by the W3C
Web Services Flow Language (WSFL), superseded by BPEL
Web template
WS-MetadataExchange
XML Interface for Network Services (XINS), provides a POX-style web service specification format

Web API

A Web API is a development in Web services where emphasis has been moving to simpler representational state transfer (REST) based communications. Restful APIs do not require XML-based Web service protocols (SOAP and WSDL) to support their interfaces.

W3C Web services
In relation to W3C Web services, the W3C defined a Web service as:

A web service is a software system designed to support interoperable machine-to-machine interaction over a network. It has an interface described in a machine-processable format (specifically WSDL). Other systems interact with the web service in a manner prescribed by its description using SOAP-messages, typically conveyed using HTTP with an XML serialization in conjunction with other web-related standards.
W3C Web Services may use SOAP over HTTP protocol, allowing less costly (more efficient) interactions over the Internet than via proprietary solutions like EDI/B2B. Besides SOAP over HTTP, Web services can also be implemented on other reliable transport mechanisms like FTP. In a 2002 document, the Web Services Architecture Working Group defined a Web services architecture, requiring a standardized implementation of a "Web service."

Explanation

The term "Web service" describes a standardized way of integrating Web-based applications using the XML, SOAP, WSDL and UDDI open standards over an Internet Protocol backbone. XML is the data format used to contain the data and provide metadata around it, SOAP is used to transfer the data, WSDL is used for describing the services available and UDDI lists what services are available.
A Web service is a method of communication between two electronic devices over a network. It is a software function provided at a network address over the Web with the service always-on as in the concept of utility computing.
Many organizations use multiple software systems for management. Different software systems often need to exchange data with each other, and a Web service is a method of communication that allows two software systems to exchange this data over the Internet. The software system that requests data is called a service requester, whereas the software system that would process the request and provide the data is called a service provider.
Different software may use different programming languages, and hence there is a need for a method of data exchange that doesn't depend upon a particular programming language. Most types of software can, however, interpret XML tags. Thus, Web services can use XML files for data exchange.
Rules for communication with different systems need to be defined, such as:

How one system can request data from another system.
Which specific parameters are needed in the data request.
What would be the structure of the data produced. (Normally, data is exchanged in XML files, and the structure of the XML file is validated against a .xsd file.)
What error messages to display when a certain rule for communication is not observed, to make troubleshooting easier.
All of these rules for communication are defined in a file called WSDL (Web Services Description Language), which has a .wsdl extension. (Proposals for Autonomous Web Services (AWS) seek to develop more flexible Web services that do not rely on strict rules.)
A directory called UDDI (Universal Description, Discovery, and Integration) defines which software system should be contacted for which type of data. So when one software system needs one particular report/data, it would go to the UDDI and find out which other systems it can contact for receiving that data. Once the software system finds out which other systems it should contact, it would then contact that system using a special protocol called SOAP (Simple Object Access Protocol). The service provider system would first validate the data request by referring to the WSDL file, and then process the request and send the data under the SOAP protocol.

Automated design methods

Automated tools can aid in the creation of a Web service. For services using WSDL, it is possible to either automatically generate WSDL for existing classes (a bottom-up model) or to generate a class skeleton given existing WSDL (a top-down model).

A developer using a bottom-up model writes implementing classes first (in some programming language) and then uses a WSDL generating tool to expose methods from these classes as a Web service. This is simpler to develop but may be harder to maintain if the original classes are subject to frequent change.
A developer using a top-down model writes the WSDL document first and then uses a code generating tool to produce the class skeleton, to be completed as necessary. This model is generally considered more difficult but can produce cleaner designs and is generally more resistant to change. As long as the message formats between the sender and receiver do not change, changes in the sender and receiver themselves do not affect the Web service. The technique is also referred to as contract first since the WSDL (or contract between sender and receiver) is the starting point.
A developer using a Subset WSDL (SWSDL) (i.e. a WSDL with the subset operation in the original WSDL) can perform Web service testing and top-down development.

Criticism
Critics of non-RESTful Web services often complain that they are too complex and based upon large software vendors or integrators, rather than typical open source implementations.
There are also concerns about performance due to Web services' use of XML as a message format and SOAP/HTTP in enveloping and transporting.

Regression testing of Web services
Functional and non-functional testing of Web services is done with the help of WSDL parsing. Regression testing is performed by identifying the changes made to upgrade software. Web service regression testing needs can be categorized in three different ways, namely, changes in WSDL, changes in the code, and selective re-testing of operations. We can capture the above three needs in three intermediate forms of Subset WSDL, namely, Difference WSDL (DWSDL), Unit WSDL (UWSDL), and Reduced WSDL (RWSDL), respectively. These three Subset WSDLs are then combined to form Combined WSDL (CWSDL) that is further used for regression testing of the Web service. This will help in Automated Web Service Change Management (AWSCM), by performing the selection of the relevant test cases to construct a reduced test suite from the old test suite.

Web services testing can also be automated using several test automation tools like SoapUI, Oracle Application Testing Suite (OATS), Unified Functional Testing, Selenium, etc.

Web service change management
Work-related to the capture and visualization of changes made to a Web service. Visualization and computation of changes can be done in the form of intermediate artifacts (Subset WSDL). The insight on the computation of change impact is helpful in testing, top-down development and reduce regression testing. AWSCM is a tool that can identify subset operations in a WSDL file to construct a subset WSDL.

Discovering and Searching for Web Services
While UDDI was intended to serve as a service directory and become the means to discovering web services, many vendors discontinued their UDDI solutions or repositories between 2005 and 2008, including Microsoft, SAP, IBM, among others. A key study published in WWW2008 Conference (Beijing, China) presented the state of SOAP-based web services and concluded that only 63% of the available SOAP-based web services at the time of the study were actually active or can be invoked. The study also found that search engines were becoming an ideal source for searching for web services compared to that of service registries like the UDDI due its design complexity.

See also
List of web service frameworks
List of web service protocols
List of web service specifications
Middleware
Service-oriented architecture (SOA)
Web Map Service
Web API

Notes

References

External links

Messaging Design Pattern Archived 2018-10-09 at the Wayback Machine documentation at SOA Patterns
The Web Services Activity page at W3C
Web Services Architecture, the W3C Working Group Note (11 February 2004)
Investigating Web Services on the World Wide Web, the analysis presented at the WWW2008 conference
Guide to Secure Web Services (SP 800–95) at NIST
