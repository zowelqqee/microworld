# Integrated services

Source: https://en.wikipedia.org/wiki/Integrated_services
Retrieved at: 2026-06-22T08:56:06Z
Revision ID: 1304231988
Raw text SHA256: b87580d874bc9ce7f8eb3455bfbadd8c9fdcdde37a922c038477d3baae756c2f
Status: LOCAL_WIKIPEDIA_SNAPSHOT
Safe for accepted memory: false
Requires ingestion/quarantine/promotion/regression: true
In computer networking, integrated services or IntServ is an architecture that specifies the elements to guarantee quality of service (QoS) on networks. IntServ can for example be used to allow video and sound to reach the receiver without interruption.
IntServ specifies a fine-grained QoS system, which is often contrasted with DiffServ's coarse-grained control system.
Under IntServ, every router in the system implements IntServ, and every application that requires some kind of QoS guarantee has to make an individual reservation. Flow specs describe what the reservation is for, while RSVP is the underlying mechanism to signal it across the network.

Flow specs
There are two parts to a flow spec:

What does the traffic look like? Done in the Traffic SPECification part, also known as TSPEC.
What guarantees does it need? Done in the service Request Specification part, also known as RSpec.
TSPECs include token bucket algorithm parameters. The idea is that there is a token bucket which slowly fills up with tokens, arriving at a constant rate. Every packet which is sent requires a token, and if there are no tokens, then it cannot be sent. Thus, the rate at which tokens arrive dictates the average rate of traffic flow, while the depth of the bucket dictates how 'bursty' the traffic is allowed to be.
TSPECs typically just specify the token rate and the bucket depth. For example, a video with a refresh rate of 75 frames per second, with each frame taking 10 packets, might specify a token rate of 750 Hz, and a bucket depth of only 10. The bucket depth would be sufficient to accommodate the 'burst' associated with sending an entire frame all at once. On the other hand, a conversation would need a lower token rate, but a much higher bucket depth. This is because there are often pauses in conversations, so they can make do with less tokens by not sending the gaps between words and sentences. However, this means the bucket depth needs to be increased to compensate for the traffic being burstier.
RSpecs specify what requirements there are for the flow: it can be normal internet 'best effort', in which case no reservation is needed. This setting is likely to be used for webpages, FTP, and similar applications. The 'Controlled Load' setting mirrors the performance of a lightly loaded network: there may be occasional glitches when two people access the same resource by chance, but generally both delay and drop rate are fairly constant at the desired rate. This setting is likely to be used by soft QoS applications. The 'Guaranteed' setting gives an absolutely bounded service, where the delay is promised to never go above a desired amount, and packets never dropped, provided the traffic stays within spec.

RSVP
The Resource Reservation Protocol (RSVP) is described in RFC 2205. All machines on the network capable of sending QoS data send a PATH message every 30 seconds, which spreads out through the networks. Those who want to listen to them send a corresponding RESV (short for "Reserve") message which then traces the path backwards to the sender. The RESV message contains the flow specs.
The routers between the sender and listener have to decide if they can support the reservation being requested, and, if they cannot, they send a reject message to let the listener know about it. Otherwise, once they accept the reservation they have to carry the traffic.
The routers then store the nature of the flow, and also police it. This is all done in soft state, so if nothing is heard for a certain length of time, then the reader will time out and the reservation will be cancelled. This solves the problem if either the sender or the receiver crash or are shut down incorrectly without first cancelling the reservation. The individual routers may, at their option, police the traffic to check that it conforms to the flow specs.

Problems
In order for IntServ to work, all routers along the traffic path must support it. Furthermore, many states must be stored in each router. As a result, IntServ works on a small-scale, but as the system scales up to larger networks or the Internet, it becomes resource intensive to track of all of the reservations.
One way to solve the scalability problem is by using a multi-level approach, where per-microflow resource reservation (such as resource reservation for individual users) is done in the edge network, while in the core network resources are reserved for aggregate flows only.  The routers that lie between these different levels must adjust the amount of aggregate bandwidth reserved from the core network so that the reservation requests for individual flows from the edge network can be better satisfied.

References

External links
RFC 1633 - Integrated Services in the Internet Architecture: an Overview
RFC 2211 - Specification of the Controlled-Load Network Element Service
RFC 2212 - Specification of Guaranteed Quality of Service
RFC 2215 - General Characterization Parameters for Integrated Service Network Elements
RFC 2205 - Resource ReSerVation Protocol (RSVP)
Cisco.com, Cisco Whitepaper about IntServ and DiffServ
