import java.net.InetSocketAddress;

import jp.seraph.jsade.core.AgentContext;


public class TutorialAgentContext implements AgentContext {

    @Override
    public String getModelInitializer() {
        return "(scene rsg/agent/nao/nao.rsg)";
    }

    @Override
    public InetSocketAddress getServerAddress() {
        return new InetSocketAddress("127.0.0.1", 3100);
    }

    @Override
    public String getTeamName() {
        return "TutorialAgent";
    }

    @Override
    public int getUniformNumber() {
        return 0;
    }
}