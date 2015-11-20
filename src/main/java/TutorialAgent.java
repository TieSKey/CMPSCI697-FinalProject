import jp.seraph.jsade.core.Agent;
import jp.seraph.jsade.core.DefaultAgentRunner;
import jp.seraph.jsade.core.DefaultWorld;
import jp.seraph.jsade.core.World;
import jp.seraph.jsade.math.Angle;
import jp.seraph.jsade.math.AngleVelocity;
import jp.seraph.jsade.model.Player;
import jp.seraph.jsade.model.nao.NaoJointIdentifier;
import jp.seraph.jsade.model.nao.NaoModelManager;


public class TutorialAgent implements Agent {
    private boolean mFlag = true;

    @Override
    public Player think(World aWorld) {
        Player tPlayer = aWorld.createPlayer();

        // 首を左右に振るだけ
        if(mFlag){
            if(tPlayer.setJointAngle(NaoJointIdentifier.HJ1, Angle.createAngleAsDegree(100)).equals(AngleVelocity.ZERO))
                mFlag = false;
        }else{
            if(tPlayer.setJointAngle(NaoJointIdentifier.HJ1, Angle.createAngleAsDegree(-100)).equals(AngleVelocity.ZERO))
                mFlag = true;
        }
        return tPlayer;
    }

    /**
     * @param args
     */
    public static void main(String[] args) {
        DefaultAgentRunner tRunner = new DefaultAgentRunner(new DefaultWorld(new NaoModelManager(0)), new TutorialAgentContext(), new TutorialAgent());

        tRunner.start();

        try{
            tRunner.join();
        }catch(InterruptedException e){
            e.printStackTrace();
        }
    }
}