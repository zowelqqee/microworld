# Post Office Protocol

Source: https://en.wikipedia.org/wiki/Post_Office_Protocol
Retrieved at: 2026-06-22T08:31:00Z
Revision ID: 1346598859
Raw text SHA256: 57ca1355c293b25c7d1527b23bcb92819140fd160f2e06105c85fc70ac5bcc11
Status: LOCAL_WIKIPEDIA_SNAPSHOT
Safe for accepted memory: false
Requires ingestion/quarantine/promotion/regression: true
In computing, the Post Office Protocol (POP) is an application-layer Internet standard protocol used by e-mail clients to retrieve e-mail from a mail server. Today, POP version 3 (POP3) is the most commonly used version. Together with IMAP, it is one of the most common protocols for email retrieval.

Purpose
The Post Office Protocol provides access via an Internet Protocol (IP) network for a user client application to a mailbox (maildrop) maintained on a mail server. The protocol supports list, retrieve and delete operations for messages. POP3 clients connect, retrieve all messages, store them on the client computer, and finally delete them from the server. This design of POP and its procedures was driven by the need of users having only temporary Internet connections, such as dial-up access, allowing these users to retrieve e-mail when connected, and subsequently to view and manipulate the retrieved messages when offline.
POP3 clients also have an option to leave mail on the server after retrieval, and in this mode of operation, clients will only download new messages which are identified by using the UIDL command (unique-id list). By contrast, the Internet Message Access Protocol (IMAP) was designed to normally leave all messages on the server to permit management with multiple client applications, and to support both connected (online) and disconnected (offline) modes of operation.
A POP3 server listens on TCP well-known port number 110 for service requests. Encrypted communication for POP3 is either requested after protocol initiation, using the STLS command, if supported, or by POP3S, which connects to the server using Transport Layer Security (TLS) or Secure Sockets Layer (SSL) on well-known TCP port number 995.
Messages available to the client are determined when a POP3 session opens the maildrop, and are identified by message-number local to that session or, optionally, by a unique identifier assigned to the message by the POP server. This unique identifier is permanent and unique to the maildrop and allows a client to access the same message in different POP sessions. Mail is retrieved and marked for deletion by the message-number. When the client exits the session, mail marked for deletion is removed from the maildrop.

History
The first version of the Post Office Protocol, POP1, was specified in RFC 918 (1984) by Joyce K. Reynolds. POP2 was specified in RFC 937 (1985).
POP3 is the version in most common use. It originated with RFC 1081 (1988) but the most recent specification is RFC 1939, updated with an extension mechanism (RFC 2449) and an authentication mechanism in RFC 1734. This led to a number of POP implementations such as Pine, POPmail, and other early mail clients.
While the original POP3 specification supported only an unencrypted USER/PASS login mechanism or Berkeley .rhosts access control, today POP3 supports several authentication methods to provide varying levels of protection against illegitimate access to a user's e-mail. Most are provided by the POP3 extension mechanisms. POP3 clients support SASL authentication methods via the AUTH extension. MIT Project Athena also produced a Kerberized version. RFC 1460 introduced APOP into the core protocol. APOP is a challenge–response protocol which uses the MD5 hash function in an attempt to avoid replay attacks and disclosure of the shared secret. Clients implementing APOP include Mozilla Thunderbird, Opera Mail, Eudora, KMail, Novell Evolution, RimArts' Becky!, Windows Live Mail, PowerMail, Apple Mail, and Mutt. RFC 1460 was obsoleted by RFC 1725, which was in turn obsoleted by RFC 1939.

POP4
POP4 exists only as an informal proposal adding basic folder management, multipart message support, as well as message flag management to compete with IMAP; however, its development has not progressed since 2003. There are now two known POP4 server implementations. As of October 2013, the POP4.org domain and website are now hosted by simbey.com, which also runs the other POP4 server implementation.

Extensions and specifications
An extension mechanism was proposed in RFC 2449 to accommodate general extensions as well as announce in an organized manner support for optional commands, such as TOP and UIDL. The RFC did not intend to encourage extensions, and reaffirmed that the role of POP3 is to provide simple support for mainly download-and-delete requirements of mailbox handling.
The extensions are termed capabilities and are listed by the CAPA command. With the exception of APOP, the optional commands were included in the initial set of capabilities. Following the lead of ESMTP (RFC 5321), capabilities beginning with an X signify local capabilities.

STARTTLS
The STARTTLS extension allows the use of Transport Layer Security (TLS) or Secure Sockets Layer (SSL) to be negotiated using the STLS command, on the standard POP3 port, rather than an alternate. Some clients and servers instead use the alternate-port method, which uses TCP port 995 (POP3S).

SDPS
Demon Internet introduced extensions to POP3 that allow multiple accounts per domain, and has become known as Standard Dial-up POP3 Service (SDPS). To access each account, the username includes the hostname, as john@hostname or john+hostname.
Google Apps uses the same method.

Kerberized Post Office Protocol
In computing, local e-mail clients can use the Kerberized Post Office Protocol (KPOP), an application-layer Internet standard protocol, to retrieve e-mail from a remote server over a TCP/IP connection. The KPOP protocol is based on the POP3 protocol – differing in that it adds Kerberos security and that it runs by default over TCP port number 1109 instead of 110. One mail server software implementation is found in the Cyrus IMAP server.

Session example
The following POP3 session dialog is an example in RFC 1939:

S:
C:
S:    +OK POP3 server ready
C:    APOP mrose c4c9334bac560ecc979e58001b3e22fb
S:    +OK mrose's maildrop has 2 messages (320 octets)
C:    STAT
S:    +OK 2 320
C:    LIST
S:    +OK 2 messages (320 octets)
S:    1 120
S:    2 200
S:    .
C:    RETR 1
S:    +OK 120 octets
S:
S:    .
C:    DELE 1
S:    +OK message 1 deleted
C:    RETR 2
S:    +OK 200 octets
S:
S:    .
C:    DELE 2
S:    +OK message 2 deleted
C:    QUIT
S:    +OK dewey POP3 server signing off (maildrop empty)
C:
S:

POP3 servers without the optional APOP command expect the client to log in with the USER and PASS commands:

C:    USER mrose
S:    +OK User accepted
C:    PASS tanstaaf
S:    +OK Pass accepted

Comparison with IMAP
The Internet Message Access Protocol (IMAP) is an alternative and more recent mailbox access protocol. The highlights of differences are:

POP is a simpler protocol, making implementation easier.
POP moves the message from the email server to the local computer, although there is usually an option in email clients to leave the messages on the email server as well. IMAP defaults to leaving the message on the email server, simply downloading a local copy.
POP treats the mailbox as a single store, and has no concept of folders
An IMAP client performs complex queries, asking the server for headers, or the bodies of specified messages, or to search for messages meeting certain criteria. Messages in the mail repository can be marked with various status flags (e.g. "deleted" or "answered") and they stay in the repository until explicitly removed by the user—which may not be until a later session. In short: IMAP is designed to permit manipulation of remote mailboxes as if they were local. Depending on the IMAP client implementation and the mail architecture desired by the system manager, the user may save messages directly on the client machine, or save them on the server, or be given the choice of doing either.
POP provides a completely static view of the current state of the mailbox, and does not provide a mechanism to show any external changes in state during the session (the client must reconnect and re-authenticate to get an updated view).
IMAP provides a dynamic view, and sends responses for external changes in state, including newly arrived messages, as well as changes made to the mailbox by other concurrently connected clients.
POP can either retrieve an entire message with the RETR command, and for servers that support it, the headers, as well as a specified number of body lines can be accessed with the TOP command.
IMAP allows clients to retrieve any of the individual MIME parts separately – for example, retrieving the plain text without retrieving attached files, or retrieving only one of many attached files.
IMAP supports flags on the server to keep track of message state: for example, whether or not the message has been read, replied to, forwarded, or deleted.
POP provides the ability to associate unique identifiers with each message for servers which support the UIDL command. This can be any string of standard visible (non-whitespace) 7-bit ASCII characters up to 70 characters.
IMAP instead provides unique numerical identifiers for each message, local to each folder, in conjunction with a folder specific UIDVALIDITY number.
The above two message identification methods (POP UIDL and IMAP UID) are not at all related unless a server implementation which supports both protocols purposely builds the POP3 UIDL string by combining the IMAP UID and UIDVALIDITY values.

Related requests for comments (RFCs)
RFC 918 – POST OFFICE PROTOCOL
RFC 937 – POST OFFICE PROTOCOL – VERSION 2
RFC 1081 – Post Office Protocol – Version 3
RFC 1939 – Post Office Protocol – Version 3 (STD 53)
RFC 1957 – Some Observations on Implementations of the Post Office Protocol (POP3)
RFC 2195 – IMAP/POP AUTHorize Extension for Simple Challenge/Response
RFC 2384 – POP URL Scheme
RFC 2449 – POP3 Extension Mechanism
RFC 2595 – Using TLS with IMAP, POP3 and ACAP
RFC 3206 – The SYS and AUTH POP Response Codes
RFC 5034 – The Post Office Protocol (POP3) Simple Authentication and Security Layer (SASL) Authentication Mechanism
RFC 8314 – Cleartext Considered Obsolete: Use of Transport Layer Security (TLS) for Email Submission and Access

See also
List of mail server software
Comparison of email clients
Comparison of mail servers
Email encryption
Internet Message Access Protocol

References

Further reading

External links
IANA port number assignments
POP3 Sequence Diagram (Archived on 6 July 2017)
